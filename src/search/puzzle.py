import math
import random
from collections.abc import Iterable
from typing import Literal, override
from search.problem import SearchProblem, Successor

# State is a 9-tuple of ints (0..8), Action is the direction moved by the blank space (0)
PuzzleState = tuple[int, ...]
PuzzleAction = Literal["Up", "Down", "Left", "Right"]
HeuristicType = Literal["manhattan", "euclidean", "none"]

GOAL_STATE: PuzzleState = (0, 1, 2, 3, 4, 5, 6, 7, 8)


class EightPuzzleProblem(SearchProblem[PuzzleState, PuzzleAction]):
    """
    8-Puzzle implementation of SearchProblem.
    Goal state is (0, 1, 2, 3, 4, 5, 6, 7, 8) where 0 is the blank tile.
    """

    def __init__(
        self,
        initial: PuzzleState = (1, 2, 5, 3, 4, 0, 6, 7, 8),
        goal: PuzzleState = GOAL_STATE,
        heuristic_type: HeuristicType = "manhattan",
    ):
        self._initial = initial
        self._goal = goal
        self._heuristic_type = heuristic_type

        self._goal_coords = {tile: (idx // 3, idx % 3) for idx, tile in enumerate(goal)}

    @override
    def initial_state(self) -> PuzzleState:
        return self._initial

    @override
    def is_goal(self, state: PuzzleState) -> bool:
        return state == self._goal

    @override
    def successors(self, state: PuzzleState) -> Iterable[Successor[PuzzleState, PuzzleAction]]:
        blank_idx = state.index(0)
        row, col = blank_idx // 3, blank_idx % 3

        moves: list[tuple[PuzzleAction, int, int]] = [
            ("Up", row - 1, col),
            ("Down", row + 1, col),
            ("Left", row, col - 1),
            ("Right", row, col + 1),
        ]

        for action, new_row, new_col in moves:
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                target_idx = new_row * 3 + new_col
                new_state_list = list(state)
                # Swap blank space (0) with adjacent tile
                new_state_list[blank_idx], new_state_list[target_idx] = (
                    new_state_list[target_idx],
                    new_state_list[blank_idx],
                )
                yield Successor(action=action, state=tuple(new_state_list), cost=1.0)

    @override
    def heuristic(self, state: PuzzleState) -> float:
        if self._heuristic_type == "none":
            return 0.0

        total_h = 0.0
        for idx, tile in enumerate(state):
            if tile == 0:
                continue
            curr_row, curr_col = idx // 3, idx % 3
            goal_row, goal_col = self._goal_coords[tile]

            if self._heuristic_type == "manhattan":
                total_h += abs(curr_row - goal_row) + abs(curr_col - goal_col)
            elif self._heuristic_type == "euclidean":
                total_h += math.sqrt((curr_row - goal_row) ** 2 + (curr_col - goal_col) ** 2)

        return total_h

    @override
    def tie_breaker(self, state: PuzzleState) -> PuzzleState:
        return state

    @staticmethod
    def is_solvable(state: PuzzleState) -> bool:
        """
        An 8-puzzle state is solvable to (0,1,2,3,4,5,6,7,8) if and only if
        the number of inversions (ignoring 0) is even.
        """
        if len(state) != 9 or set(state) != set(range(9)):
            return False
        tiles = [t for t in state if t != 0]
        inversions = 0
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    inversions += 1
        return inversions % 2 == 0

    @staticmethod
    def random_solvable_state(steps: int = 25) -> PuzzleState:
        """Generates a guaranteed solvable state by walking backwards from the goal."""
        problem = EightPuzzleProblem(initial=GOAL_STATE)
        curr = GOAL_STATE
        for _ in range(steps):
            succs = list(problem.successors(curr))
            curr = random.choice(succs).state
        return curr