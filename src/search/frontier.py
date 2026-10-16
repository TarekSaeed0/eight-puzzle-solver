from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, override
from collections.abc import Callable, Iterable
from collections import deque
from search.node import Node


S = TypeVar("S")
A = TypeVar("A")


class Frontier(ABC, Generic[S, A]):
    @abstractmethod
    def push(self, node: Node[S, A]) -> None:
        pass

    @abstractmethod
    def pop(self) -> Node[S, A]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def nodes(self) -> Iterable[Node[S, A]]:
        pass


class StackFrontier(Frontier[S, A]):
    queue: deque[Node[S, A]]

    def __init__(self):
        self.queue = deque()

    @override
    def push(self, node: Node[S, A]) -> None:
        self.queue.append(node)

    @override
    def pop(self) -> Node[S, A]:
        return self.queue.pop()

    @override
    def is_empty(self) -> bool:
        return len(self.queue) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        return self.queue


class QueueFrontier(Frontier[S, A]):
    queue: deque[Node[S, A]]

    def __init__(self):
        self.queue = deque()

    @override
    def push(self, node: Node[S, A]) -> None:
        self.queue.append(node)

    @override
    def pop(self) -> Node[S, A]:
        return self.queue.popleft()

    @override
    def is_empty(self) -> bool:
        return len(self.queue) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        return self.queue


class PriorityQueueFrontier(Frontier[S, A]):
    key: Callable[[Node[S, A]], Any]
    heap: list[Node[S, A]]
    state_to_index: dict[S, int]

    def __init__(self, key: Callable[[Node[S, A]], Any]):
        self.key = key
        self.heap = []
        self.state_to_index = {}

    @override
    def push(self, node: Node[S, A]) -> None:
        if node.state in self.state_to_index:
            self.replace(node)
            return

        i = len(self.heap)
        self.heap.append(node)
        self.state_to_index[node.state] = i
        self._sift_up(i)

    @override
    def pop(self) -> Node[S, A]:
        if self.is_empty():
            raise IndexError("pop from empty priority queue")

        root = self.heap[0]
        del self.state_to_index[root.state]

        last = self.heap.pop()
        if self.heap:
            self.heap[0] = last
            self.state_to_index[last.state] = 0
            self._sift_down(0)

        return root

    def get(self, state: S) -> Node[S, A] | None:
        i = self.state_to_index.get(state)
        if i is not None:
            return self.heap[i]

        return None

    def replace(self, node: Node[S, A]) -> None:
        i = self.state_to_index.get(node.state)
        if i is None:
            self.push(node)
            return

        old_node = self.heap[i]
        self.heap[i] = node

        if self.key(node) < self.key(old_node):
            self._sift_up(i)
        else:
            self._sift_down(i)

    @override
    def is_empty(self) -> bool:
        return len(self.heap) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        return self.heap

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = self._parent(i)
            if self.key(self.heap[i]) < self.key(self.heap[parent]):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        while True:
            left = self._left(i)
            right = self._right(i)
            smallest = i

            if left < len(self.heap) and self.key(self.heap[left]) < self.key(
                self.heap[smallest]
            ):
                smallest = left
            if right < len(self.heap) and self.key(self.heap[right]) < self.key(
                self.heap[smallest]
            ):
                smallest = right

            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.state_to_index[self.heap[i].state] = i
        self.state_to_index[self.heap[j].state] = j
