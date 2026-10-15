from dataclasses import dataclass
from typing import Generic, TypeVar
from search.problem import SearchProblem


S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Node(Generic[S, A]):
    state: S
    parent: "Node[S, A] | None" = None
    action: A | None = None
    path_cost: float = 0
    heuristic: float = 0

    def children(self, problem: "SearchProblem[S, A]") -> list["Node[S, A]"]:
        return [
            Node(
                state=s.state,
                parent=self,
                action=s.action,
                path_cost=self.path_cost + s.cost,
                heuristic=problem.heuristic(s.state),
            )
            for s in problem.successors(self.state)
        ]

    def reconstruct_path(self) -> list[tuple[A | None, S]]:
        node, path = self, []
        while node:
            path.append((node.action, node.state))
            node = node.parent
        path.reverse()
        return path
