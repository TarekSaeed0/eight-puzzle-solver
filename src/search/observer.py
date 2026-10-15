from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from search.event import SearchEvent

S = TypeVar("S")
A = TypeVar("A")


class SearchObserver(ABC, Generic[S, A]):
    @abstractmethod
    def on_event(self, event: SearchEvent[S, A]) -> None:
        pass
