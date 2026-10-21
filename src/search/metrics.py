import time
from typing import override
from search.event import (
    NodeExpandedEvent,
    NodePushedEvent,
    NodeReplacedEvent,
    SearchEndedEvent,
    SearchEvent,
    SearchStartedEvent,
)
from search.observer import SearchObserver
from search.puzzle import PuzzleAction, PuzzleState


class PuzzleMetricsObserver(SearchObserver[PuzzleState, PuzzleAction]):


    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.nodes_expanded: int = 0
        self.max_search_depth: int = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.running_time_ms: float = 0.0
        self.goal_depth: int = 0
        self.path_cost: float = 0.0
        self.path: list[tuple[PuzzleAction | None, PuzzleState]] = []
        self.solved: bool = False
        self._timer_started: bool = False

    @override
    def on_event(self, event: SearchEvent[PuzzleState, PuzzleAction]) -> None:
        if isinstance(event, SearchStartedEvent):
            if not self._timer_started:
                self.start_time = time.perf_counter()
                self._timer_started = True

        elif isinstance(event, NodeExpandedEvent):
            self.nodes_expanded += 1
            if event.node.depth > self.max_search_depth:
                self.max_search_depth = event.node.depth

        elif isinstance(event, NodePushedEvent):
            if event.node.depth > self.max_search_depth:
                self.max_search_depth = event.node.depth

        elif isinstance(event, NodeReplacedEvent):
            if event.new_node.depth > self.max_search_depth:
                self.max_search_depth = event.new_node.depth

        elif isinstance(event, SearchEndedEvent):
            self.end_time = time.perf_counter()
            self.running_time_ms = (self.end_time - self.start_time) * 1000.0
            if event.final_node is not None:
                self.solved = True
                self.goal_depth = event.final_node.depth
                self.path_cost = event.final_node.path_cost
                self.path = event.final_node.reconstruct_path()