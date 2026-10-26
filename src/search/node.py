from dataclasses import dataclass
from typing import Generic, TypeVar

from search.problem import SearchProblem

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Node(Generic[S, A]):
    """A state in the search tree and its route back to the root."""

    state: S
    parent: "Node[S, A] | None" = None
    action: A | None = None
    path_cost: float = 0
    depth: int = 0

    def children(self, problem: "SearchProblem[S, A]") -> list["Node[S, A]"]:
        """Create child nodes from all valid successors of this state."""
        return [
            Node(
                state=s.state,
                parent=self,
                action=s.action,
                path_cost=self.path_cost + s.cost,
                depth=self.depth + 1,
            )
            for s in problem.successors(self.state)
        ]

    def reconstruct_path(self) -> list[tuple[A | None, S]]:
        """Return the root-to-this-node path as action/state pairs."""
        node, path = self, []
        while node:
            path.append((node.action, node.state))
            node = node.parent
        path.reverse()
        return path
