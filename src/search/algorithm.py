from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar, override

from search.event import (
    NodeExpandedEvent,
    NodePoppedEvent,
    NodePushedEvent,
    NodeReplacedEvent,
    NodeSkippedEvent,
    SearchEndedEvent,
    SearchEvent,
    SearchStartedEvent,
)
from search.frontier import (
    Frontier,
    PriorityQueueFrontier,
    QueueFrontier,
    StackFrontier,
)
from search.node import Node
from search.observer import SearchObserver
from search.problem import SearchProblem

S = TypeVar("S")
A = TypeVar("A")


def graph_search(
    problem: SearchProblem[S, A],
    frontier: Frontier[S, A],
    emit: Callable[[SearchEvent[S, A]], None],
) -> Node[S, A] | None:
    """Run graph search with the supplied frontier implementation.

    States are expanded at most once, and ``emit`` receives lifecycle events
    as nodes move through the frontier.
    """
    # A state is expanded at most once; duplicate nodes can still remain in
    # the frontier because they may have been queued before the state was seen.
    explored = set()

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(frontier.nodes(), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(frontier.nodes(), explored, current_node))

        if current_node.state in explored:
            emit(NodeSkippedEvent(frontier.nodes(), explored, current_node))

            continue

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(frontier.nodes(), explored, current_node))

            return current_node

        explored.add(current_node.state)

        children = current_node.children(problem)

        for child in children:
            if child.state in explored:
                continue

            frontier.push(child)

            emit(NodePushedEvent(frontier.nodes(), explored, child))

        emit(
            NodeExpandedEvent(frontier.nodes(), explored, current_node, list(children))
        )

    emit(SearchEndedEvent(frontier.nodes(), explored, None))

    return None


def graph_search_depth_limited(
    problem: SearchProblem[S, A],
    depth_limit: int,
    emit: Callable[[SearchEvent[S, A]], None],
) -> tuple[Node[S, A] | None, bool]:
    """Run depth-limited search and report whether the limit caused a cutoff."""
    frontier = StackFrontier[S, A]()
    explored = set()

    # Keep the shallowest depth seen for each state. A deeper occurrence cannot
    # produce a better result within the same depth-limited iteration.
    best_depth: dict[S, int] = {}

    has_cutoff_occurred = False

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(frontier.nodes(), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(frontier.nodes(), explored, current_node))

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(frontier.nodes(), explored, current_node))

            return current_node, False

        seen_depth = best_depth.get(current_node.state)
        if seen_depth is not None and seen_depth <= current_node.depth:
            emit(NodeSkippedEvent(frontier.nodes(), explored, current_node))

            continue

        best_depth[current_node.state] = current_node.depth
        explored.add(current_node.state)

        if current_node.depth >= depth_limit:
            # A cutoff means that there is still nodes we haven't expanded which, means we should continue iterating.
            has_cutoff_occurred = True
            continue

        children = current_node.children(problem)

        for child in children:
            seen_depth = best_depth.get(child.state)
            if seen_depth is not None and seen_depth <= child.depth:
                continue

            frontier.push(child)

            emit(NodePushedEvent(frontier.nodes(), explored, child))

        emit(
            NodeExpandedEvent(frontier.nodes(), explored, current_node, list(children))
        )

    emit(SearchEndedEvent(frontier.nodes(), explored, None))

    return None, has_cutoff_occurred


def graph_search_with_replacement(
    problem: SearchProblem[S, A],
    frontier: PriorityQueueFrontier[S, A],
    emit: Callable[[SearchEvent[S, A]], None],
) -> Node[S, A] | None:
    """Run cost-based search with replacement of more expensive queued nodes."""
    explored = set()

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(frontier.nodes(), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(frontier.nodes(), explored, current_node))

        if current_node.state in explored:
            emit(NodeSkippedEvent(frontier.nodes(), explored, current_node))

            continue

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(frontier.nodes(), explored, current_node))

            return current_node

        explored.add(current_node.state)

        children = current_node.children(problem)

        for child in children:
            if child.state in explored:
                continue

            # If the child state is already in the frontier, we need to check if the new path
            # cost is lower than the existing one. If it is, we replace the existing node with
            # the new one. Otherwise, we simply push the new child node onto the frontier.
            existing_node = frontier.get(child.state)
            if not existing_node:
                frontier.push(child)

                emit(NodePushedEvent(frontier.nodes(), explored, child))
            elif child.path_cost < existing_node.path_cost:
                frontier.replace(child)

                emit(
                    NodeReplacedEvent(frontier.nodes(), explored, existing_node, child)
                )

        emit(
            NodeExpandedEvent(frontier.nodes(), explored, current_node, list(children))
        )

    emit(SearchEndedEvent(frontier.nodes(), explored, None))

    return None


class SearchAlgorithm(ABC, Generic[S, A]):
    """Base interface for a search algorithm."""

    @abstractmethod
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Return a goal node, or ``None`` if no solution exists."""


class ObservableSearchAlgorithm(SearchAlgorithm[S, A]):
    """Search base class that broadcasts events to registered observers."""

    def __init__(self):
        """Create an algorithm with no observers."""
        self.observers: list[SearchObserver[S, A]] = []

    def add_observer(self, observer: SearchObserver[S, A]) -> None:
        """Register an observer for future search events."""
        self.observers.append(observer)

    def remove_observer(self, observer: SearchObserver[S, A]) -> None:
        """Unregister an observer."""
        self.observers.remove(observer)

    def emit(self, event: SearchEvent[S, A]) -> None:
        """Deliver one event to every registered observer."""
        for observer in self.observers:
            observer.on_event(event)


class DepthFirstSearchAlgorithm(ObservableSearchAlgorithm[S, A]):
    """Explore the newest frontier node first."""

    @override
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Search using a LIFO frontier."""
        return graph_search(problem, StackFrontier[S, A](), self.emit)


class BreadthFirstSearchAlgorithm(ObservableSearchAlgorithm[S, A]):
    """Explore nodes in increasing depth order."""

    @override
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Search using a FIFO frontier."""
        return graph_search(problem, QueueFrontier[S, A](), self.emit)


class IterativeDeepeningDepthFirstSearchAlgorithm(ObservableSearchAlgorithm[S, A]):
    """Repeat depth-limited DFS with an increasing depth limit."""

    @override
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Repeat depth-limited searches until a goal or exhaustion."""
        depth_limit = 0
        while True:
            final_node, has_cutoff_occurred = graph_search_depth_limited(
                problem, depth_limit, self.emit
            )

            if final_node is not None:
                return final_node

            if not has_cutoff_occurred:
                return None

            depth_limit += 1


class UniformCostSearchAlgorithm(ObservableSearchAlgorithm[S, A]):
    """Expand the node with the lowest path cost first."""

    @override
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Search in order of accumulated path cost."""
        return graph_search_with_replacement(
            problem,
            PriorityQueueFrontier[S, A](
                key=lambda n: (n.path_cost, problem.tie_breaker(n.state))
            ),
            self.emit,
        )


class AStarSearchAlgorithm(ObservableSearchAlgorithm[S, A]):
    """Order nodes by path cost plus the problem's heuristic estimate."""

    @override
    def search(self, problem: SearchProblem[S, A]) -> Node[S, A] | None:
        """Search in order of accumulated cost plus heuristic estimate."""
        return graph_search_with_replacement(
            problem,
            PriorityQueueFrontier[S, A](
                key=lambda n: (
                    n.path_cost + problem.heuristic(n.state),
                    problem.tie_breaker(n.state),
                )
            ),
            self.emit,
        )
