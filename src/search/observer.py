from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from search.event import SearchEvent

S = TypeVar("S")
A = TypeVar("A")


class SearchObserver(ABC, Generic[S, A]):
    """Consumer interface for events emitted during a search."""

    @abstractmethod
    def on_event(self, event: SearchEvent[S, A]) -> None:
        """Handle one search event."""
