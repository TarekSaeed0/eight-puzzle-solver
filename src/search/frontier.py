from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, override

from search.node import Node

S = TypeVar("S")
A = TypeVar("A")


class Frontier(ABC, Generic[S, A]):
    """Collection of pending nodes that determines search order."""

    @abstractmethod
    def push(self, node: Node[S, A]) -> None:
        """Add a node to the frontier."""
        pass

    @abstractmethod
    def pop(self) -> Node[S, A]:
        """Remove and return the next node to search."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Return whether no nodes are pending."""
        pass

    @abstractmethod
    def nodes(self) -> Iterable[Node[S, A]]:
        """Return pending nodes for inspection and search events."""
        pass


class StackFrontier(Frontier[S, A]):
    """LIFO frontier used by depth-first search."""

    queue: deque[Node[S, A]]

    def __init__(self):
        """Create an empty stack frontier."""
        self.queue = deque()

    @override
    def push(self, node: Node[S, A]) -> None:
        """Add ``node`` to the top of the stack."""
        self.queue.append(node)

    @override
    def pop(self) -> Node[S, A]:
        """Remove and return the most recently added node."""
        return self.queue.pop()

    @override
    def is_empty(self) -> bool:
        """Return whether the stack contains no nodes."""
        return len(self.queue) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        """Return the nodes currently waiting in the stack."""
        return self.queue


class QueueFrontier(Frontier[S, A]):
    """FIFO frontier used by breadth-first search."""

    queue: deque[Node[S, A]]

    def __init__(self):
        """Create an empty queue frontier."""
        self.queue = deque()

    @override
    def push(self, node: Node[S, A]) -> None:
        """Append ``node`` to the end of the queue."""
        self.queue.append(node)

    @override
    def pop(self) -> Node[S, A]:
        """Remove and return the earliest added node."""
        return self.queue.popleft()

    @override
    def is_empty(self) -> bool:
        """Return whether the queue contains no nodes."""
        return len(self.queue) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        """Return the nodes currently waiting in the queue."""
        return self.queue


class PriorityQueueFrontier(Frontier[S, A]):
    """Min-heap frontier with lookup and replacement by state."""

    key: Callable[[Node[S, A]], Any]
    heap: list[Node[S, A]]
    state_to_index: dict[S, int]

    def __init__(self, key: Callable[[Node[S, A]], Any]):
        """Create a frontier ordered by ``key`` applied to each node."""
        self.key = key
        self.heap = []
        self.state_to_index = {}

    @override
    def push(self, node: Node[S, A]) -> None:
        """Insert ``node`` and restore the min-heap ordering."""
        if node.state in self.state_to_index:
            self.replace(node)
            return

        i = len(self.heap)
        self.heap.append(node)
        self.state_to_index[node.state] = i
        self._sift_up(i)

    @override
    def pop(self) -> Node[S, A]:
        """Remove and return the node with the smallest priority key."""
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
        """Return the queued node for ``state``, if present."""
        i = self.state_to_index.get(state)
        if i is not None:
            return self.heap[i]

        return None

    def replace(self, node: Node[S, A]) -> None:
        """Replace a queued state and restore heap ordering."""
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
        """Return whether the heap contains no nodes."""
        return len(self.heap) == 0

    @override
    def nodes(self) -> Iterable[Node[S, A]]:
        """Return the nodes currently stored in the heap."""
        return self.heap

    @staticmethod
    def _parent(i: int) -> int:
        """Return the parent index for heap index ``i``."""
        return (i - 1) // 2

    @staticmethod
    def _left(i: int) -> int:
        """Return the left-child index for heap index ``i``."""
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        """Return the right-child index for heap index ``i``."""
        return 2 * i + 2

    def _sift_up(self, i: int) -> None:
        """Move a node toward the root until the heap property is restored."""
        # Restore the heap property after inserting or lowering a key.
        while i > 0:
            parent = self._parent(i)
            if self.key(self.heap[i]) < self.key(self.heap[parent]):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Move a node toward the leaves until the heap property is restored."""
        # Restore the heap property after removing or raising a key.
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
        """Swap two heap entries and update their index records."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.state_to_index[self.heap[i].state] = i
        self.state_to_index[self.heap[j].state] = j
