from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar
from search.node import Node

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class SearchEvent(ABC, Generic[S, A]):
    """Base event containing the frontier and explored-state snapshot."""

    frontier: Iterable[Node[S, A]]
    explored: set[S]


@dataclass
class SearchStartedEvent(SearchEvent[S, A]):
    """Emitted after the initial node enters the frontier."""

    initial_node: Node[S, A]


@dataclass
class SearchEndedEvent(SearchEvent[S, A]):
    """Emitted when a goal is found or the frontier is exhausted."""

    final_node: Node[S, A] | None


@dataclass
class NodePoppedEvent(SearchEvent[S, A]):
    """Emitted when a node is removed for consideration."""

    node: Node[S, A]


@dataclass
class NodePushedEvent(SearchEvent[S, A]):
    """Emitted when a newly discovered node enters the frontier."""

    node: Node[S, A]


@dataclass
class NodeReplacedEvent(SearchEvent[S, A]):
    """Emitted when a cheaper route replaces a queued node."""

    old_node: Node[S, A]
    new_node: Node[S, A]


@dataclass
class NodeSkippedEvent(SearchEvent[S, A]):
    """Emitted when a node is ignored."""

    node: Node[S, A]


@dataclass
class NodeExpandedEvent(SearchEvent[S, A]):
    """Emitted after a node generates its successors."""

    node: Node[S, A]
    children: list[Node[S, A]]
