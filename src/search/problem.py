from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Successor(Generic[S, A]):
    """A state reachable by one action and its transition cost."""

    action: A
    state: S
    cost: float


class SearchProblem(ABC, Generic[S, A]):
    """Interface describing a state space that search algorithms can explore."""

    @abstractmethod
    def initial_state(self) -> S:
        """Return the state from which search starts."""

    @abstractmethod
    def is_goal(self, state: S) -> bool:
        """Return whether ``state`` satisfies the goal condition."""

    @abstractmethod
    def successors(self, state: S) -> Iterable[Successor[S, A]]:
        """Yield the legal one-step transitions from ``state``."""

    def heuristic(self, state: S) -> float:
        """Estimate remaining cost; zero provides uninformed search."""
        return 0

    def tie_breaker(self, state: S) -> Any:
        """Return a deterministic secondary priority for equal-cost nodes."""
        return 0
