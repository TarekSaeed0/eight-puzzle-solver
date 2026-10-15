from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Successor(Generic[S, A]):
    action: A
    state: S
    cost: float


class SearchProblem(ABC, Generic[S, A]):
    @abstractmethod
    def initial_state(self) -> S:
        pass

    @abstractmethod
    def is_goal(self, state: S) -> bool:
        pass

    @abstractmethod
    def successors(self, state: S) -> Iterable[Successor[S, A]]:
        pass

    def heuristic(self, state: S) -> float:
        return 0

    def tie_breaker(self, state: S) -> Any:
        return 0
