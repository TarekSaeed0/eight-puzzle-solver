from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar
from search.node import Node

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class SearchEvent(ABC, Generic[S, A]):
    frontier: list[Node[S, A]]
    explored: set[S]


@dataclass
class SearchStartedEvent(SearchEvent[S, A]):
    initial_node: Node[S, A]


@dataclass
class SearchEndedEvent(SearchEvent[S, A]):
    final_node: Node[S, A] | None


@dataclass
class NodePoppedEvent(SearchEvent[S, A]):
    node: Node[S, A]


@dataclass
class NodePushedEvent(SearchEvent[S, A]):
    node: Node[S, A]


@dataclass
class NodeReplacedEvent(SearchEvent[S, A]):
    old_node: Node[S, A]
    new_node: Node[S, A]


@dataclass
class NodeSkippedEvent(SearchEvent[S, A]):
    node: Node[S, A]


@dataclass
class NodeExpandedEvent(SearchEvent[S, A]):
    node: Node[S, A]
    children: list[Node[S, A]]
