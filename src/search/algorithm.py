from abc import ABC, abstractmethod
from typing import TypeVar
from search.node import Node
from search.event import (
    SearchEvent,
    SearchStartedEvent,
    SearchEndedEvent,
    NodePoppedEvent,
    NodePushedEvent,
    NodeReplacedEvent,
    NodeSkippedEvent,
    NodeExpandedEvent,
)
from search.observer import SearchObserver
from search.problem import SearchProblem
from search.frontier import (
    Frontier,
    StackFrontier,
    QueueFrontier,
    PriorityQueueFrontier,
)

S = TypeVar("S")
A = TypeVar("A")


def graph_search(
    problem: SearchProblem[S, A],
    frontier: Frontier[S, A],
    observer: SearchObserver[S, A] | None = None,
) -> Node[S, A] | None:
    def emit(event: SearchEvent[S, A]) -> None:
        if observer:
            observer.on_event(event)

    explored = set()

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(list(frontier.nodes()), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(list(frontier.nodes()), explored, current_node))

        if current_node.state in explored:
            emit(NodeSkippedEvent(list(frontier.nodes()), explored, current_node))

            continue

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(list(frontier.nodes()), explored, current_node))

            return current_node

        explored.add(current_node.state)

        children = current_node.children(problem)

        for child in children:
            if child.state in explored:
                continue

            frontier.push(child)

            emit(NodePushedEvent(list(frontier.nodes()), explored, child))

        emit(
            NodeExpandedEvent(
                list(frontier.nodes()), explored, current_node, list(children)
            )
        )

    emit(SearchEndedEvent(list(frontier.nodes()), explored, None))

    return None


def graph_search_depth_limited(
    problem: SearchProblem[S, A],
    depth_limit: int,
    observer: SearchObserver[S, A] | None = None,
) -> tuple[Node[S, A] | None, bool]:
    def emit(event: SearchEvent[S, A]) -> None:
        if observer:
            observer.on_event(event)

    frontier = StackFrontier[S, A]()
    explored = set()

    has_cutoff_occurred = False

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(list(frontier.nodes()), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(list(frontier.nodes()), explored, current_node))

        if current_node.state in explored:
            emit(NodeSkippedEvent(list(frontier.nodes()), explored, current_node))

            continue

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(list(frontier.nodes()), explored, current_node))

            return current_node, False

        explored.add(current_node.state)

        if current_node.depth >= depth_limit:
            has_cutoff_occurred = True
            continue

        children = current_node.children(problem)

        for child in children:
            if child.state in explored:
                continue

            frontier.push(child)

            emit(NodePushedEvent(list(frontier.nodes()), explored, child))

        emit(
            NodeExpandedEvent(
                list(frontier.nodes()), explored, current_node, list(children)
            )
        )

    emit(SearchEndedEvent(list(frontier.nodes()), explored, None))

    return None, has_cutoff_occurred


def graph_search_with_replacement(
    problem: SearchProblem[S, A],
    frontier: PriorityQueueFrontier[S, A],
    observer: SearchObserver[S, A] | None = None,
) -> Node[S, A] | None:
    def emit(event: SearchEvent[S, A]) -> None:
        if observer:
            observer.on_event(event)

    explored = set()

    initial_node = Node(state=problem.initial_state())

    frontier.push(initial_node)

    emit(SearchStartedEvent(list(frontier.nodes()), explored, initial_node))

    while not frontier.is_empty():
        current_node = frontier.pop()

        emit(NodePoppedEvent(list(frontier.nodes()), explored, current_node))

        if current_node.state in explored:
            emit(NodeSkippedEvent(list(frontier.nodes()), explored, current_node))

            continue

        if problem.is_goal(current_node.state):
            emit(SearchEndedEvent(list(frontier.nodes()), explored, current_node))

            return current_node

        explored.add(current_node.state)

        children = current_node.children(problem)

        for child in children:
            if child.state in explored:
                continue

            existing_node = frontier.get(child.state)
            if not existing_node:
                frontier.push(child)

                emit(NodePushedEvent(list(frontier.nodes()), explored, child))
            elif child.path_cost < existing_node.path_cost:
                frontier.replace(child)

                emit(
                    NodeReplacedEvent(
                        list(frontier.nodes()), explored, existing_node, child
                    )
                )

        emit(
            NodeExpandedEvent(
                list(frontier.nodes()), explored, current_node, list(children)
            )
        )

    emit(SearchEndedEvent(list(frontier.nodes()), explored, None))

    return None


class SearchAlgorithm(ABC):
    @abstractmethod
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        pass


class DepthFirstSearchAlgorithm(SearchAlgorithm):
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        return graph_search(problem, StackFrontier[S, A](), observer)


class BreadthFirstSearchAlgorithm(SearchAlgorithm):
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        return graph_search(problem, QueueFrontier[S, A](), observer)


class IterativeDeepeningDepthFirstSearchAlgorithm(SearchAlgorithm):
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        depth_limit = 0
        while True:
            final_node, has_cutoff_occurred = graph_search_depth_limited(
                problem, depth_limit, observer
            )

            if final_node is not None:
                return final_node

            if not has_cutoff_occurred:
                return None

            depth_limit += 1


class UniformCostSearchAlgorithm(SearchAlgorithm):
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        return graph_search_with_replacement(
            problem,
            PriorityQueueFrontier[S, A](
                key=lambda n: (n.path_cost, problem.tie_breaker(n.state))
            ),
            observer,
        )


class AStarSearchAlgorithm(SearchAlgorithm):
    def search(
        self, problem: SearchProblem[S, A], observer: SearchObserver[S, A] | None = None
    ) -> Node[S, A] | None:
        return graph_search_with_replacement(
            problem,
            PriorityQueueFrontier[S, A](
                key=lambda n: (
                    n.path_cost + problem.heuristic(n.state),
                    problem.tie_breaker(n.state),
                )
            ),
            observer,
        )
