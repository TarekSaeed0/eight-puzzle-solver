"""8-Puzzle Solver - Flet UI.

Design notes
------------
* One piece of state (`AppState`) is the single source of truth. Every handler
  mutates it and then calls `refresh()` / `render_results()`, which derive ALL
  visuals (tiles, enabled/disabled buttons, visibility) from that state.
* Searches run in a worker thread (`asyncio.to_thread`), so the UI never freezes
  and can show a loading indicator. They can be cancelled through an observer
  that raises `SearchCancelled` on the next search event.
* The trace is paginated (10 steps per page). DFS can legitimately return paths
  with 100,000+ steps; one widget per step would freeze or crash the UI.
* No `expand=True` inside the scrolling page (that caused the endless gray
  area): every scrollable region has either a fixed height or none at all.
"""

import asyncio
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import flet as ft
import re
from typing import Literal

from search.puzzle import EightPuzzleProblem, PuzzleAction, PuzzleState
from search.algorithm import (
    AStarSearchAlgorithm,
    BreadthFirstSearchAlgorithm,
    DepthFirstSearchAlgorithm,
    IterativeDeepeningDepthFirstSearchAlgorithm,
)
from search.event import SearchEvent
from search.metrics import PuzzleMetricsObserver
from search.observer import SearchObserver

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_BOARD: PuzzleState = (1, 2, 5, 3, 4, 0, 6, 7, 8)
TILE = 84  # px, 3 tiles + gaps = 268 px -> fits a 320 px wide phone
PAGE_SIZE = 10  # trace rows per page
MAX_PATH_PREVIEW = 300  # moves shown in "Path to goal" before truncating
MAX_CONSOLE_TRACE = 200  # print the full trace to the console up to this many steps

ARROWS = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→"}

# key -> (display label, algorithm class, heuristic type)
ALGORITHMS = {
    "BFS": ("BFS", BreadthFirstSearchAlgorithm, "none"),
    "DFS": ("DFS", DepthFirstSearchAlgorithm, "none"),
    "IDDFS": ("Iterative DFS", IterativeDeepeningDepthFirstSearchAlgorithm, "none"),
    "ASTAR_MANHATTAN": ("A* Manhattan", AStarSearchAlgorithm, "manhattan"),
    "ASTAR_EUCLIDEAN": ("A* Euclidean", AStarSearchAlgorithm, "euclidean"),
}
ALGO_OPTIONS = {
    "BFS": "BFS (Breadth-First Search)",
    "DFS": "DFS (Depth-First Search)",
    "IDDFS": "Iterative Deepening DFS",
    "ASTAR_MANHATTAN": "A* (Manhattan heuristic)",
    "ASTAR_EUCLIDEAN": "A* (Euclidean heuristic)",
}
SPEEDS = {"slow": 0.8, "normal": 0.45, "fast": 0.15, "turbo": 0.03}


# --------------------------------------------------------------------------- #
# Search plumbing (UI independent)
# --------------------------------------------------------------------------- #
class SearchCancelled(Exception):
    """Raised inside the worker thread when the user presses Cancel."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class CancelObserver(SearchObserver[PuzzleState, PuzzleAction]):
    """Aborts the running search at the next event once cancellation is requested."""

    def __init__(self, token: CancelToken) -> None:
        self.token = token

    def on_event(self, event: SearchEvent[PuzzleState, PuzzleAction]) -> None:
        if self.token.cancelled:
            raise SearchCancelled


@dataclass
class RunResult:
    key: str
    label: str
    solved: bool
    path: list[tuple[PuzzleAction | None, PuzzleState]]
    cost: float
    expanded: int
    goal_depth: int
    max_depth: int
    time_ms: float

    @property
    def moves(self) -> list[str]:
        return [a for a, _ in self.path if a is not None]


def execute_search(key: str, initial: PuzzleState, token: CancelToken) -> RunResult:
    """Run one algorithm to completion. Called from a worker thread."""
    label, factory, heuristic = ALGORITHMS[key]
    algorithm = factory()
    metrics = PuzzleMetricsObserver()
    algorithm.add_observer(metrics)
    algorithm.add_observer(CancelObserver(token))
    algorithm.search(EightPuzzleProblem(initial=initial, heuristic_type=heuristic))  # type: ignore[arg-type]
    return RunResult(
        key=key,
        label=label,
        solved=metrics.solved,
        path=metrics.path,
        cost=metrics.path_cost,
        expanded=metrics.nodes_expanded,
        goal_depth=metrics.goal_depth,
        max_depth=metrics.max_search_depth,
        time_ms=metrics.running_time_ms,
    )


def format_board_ascii(state: PuzzleState) -> str:
    """Formats a 9-tuple into a clean 3x3 text grid for traceable output."""
    lines = []
    for r in range(3):
        row_vals = [
            str(state[r * 3 + c]) if state[r * 3 + c] != 0 else " " for c in range(3)
        ]
        lines.append(" | ".join(row_vals))
    return "\n---------\n".join(lines)


def format_board_compact(state: PuzzleState) -> str:
    """One-line board for the trace list: '1 2 5 | 3 4 _ | 6 7 8'."""
    cells = ["_" if v == 0 else str(v) for v in state]
    return " | ".join(" ".join(cells[r * 3 : r * 3 + 3]) for r in range(3))


def format_time(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.2f} ms"


def build_trace_text(result: RunResult, initial: PuzzleState) -> str:
    """Complete step-by-step trace (every step, 3x3 board) as text."""
    lines = [
        f"Algorithm       : {result.label}",
        f"Initial state   : {','.join(map(str, initial))}",
        f"Cost of path    : {int(result.cost)}",
        f"Nodes expanded  : {result.expanded}",
        f"Search depth    : goal={result.goal_depth}, max={result.max_depth}",
        f"Running time    : {format_time(result.time_ms)}",
        "",
    ]
    for i, (action, state) in enumerate(result.path):
        title = "Initial state" if action is None else f"Move {action}"
        lines.append(f"Step {i}: {title}")
        lines.append(format_board_ascii(state))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing and validation of the board text typed by the user (UI independent)
# --------------------------------------------------------------------------- #

Level = Literal["empty", "incomplete", "error", "ok"]


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing: `state` is set only when level == "ok"."""

    level: Level
    message: str
    state: PuzzleState | None = None


def parse_board(text: str) -> ParseResult:
    """Parse '1,2,5,3,4,0,6,7,8', '1 2 5 3 4 0 6 7 8' or '125340678'.

    "incomplete" (fewer than 9 values) is separated from "error" so the UI can
    stay calm while the user is still typing instead of flashing red.
    """
    raw = text.strip()
    if not raw:
        return ParseResult("empty", "Enter the 9 tile values (0 = blank).")

    # Accept separators (comma / space / semicolon) or a bare string of digits.
    if re.search(r"[,;\s]", raw):
        tokens = [t for t in re.split(r"[,;\s]+", raw) if t]
    else:
        tokens = list(raw)

    for tok in tokens:
        if not (tok.isascii() and tok.isdigit()) or int(tok) > 8:
            return ParseResult(
                "error", f"Only the digits 0-8 are allowed (found '{tok}')."
            )

    nums = [int(t) for t in tokens]

    repeated = sorted({n for n in nums if nums.count(n) > 1})
    if repeated:
        listed = ", ".join(map(str, repeated))
        return ParseResult(
            "error", f"Each digit must appear exactly once (repeated: {listed})."
        )

    if len(nums) < 9:
        return ParseResult("incomplete", f"{len(nums)} of 9 values entered...")
    if len(nums) > 9:
        return ParseResult(
            "error", f"Too many values: found {len(nums)}, need exactly 9."
        )

    state: PuzzleState = tuple(nums)
    if not EightPuzzleProblem.is_solvable(state):
        return ParseResult(
            "error",
            "Unsolvable board (odd number of inversions). Swap any two tiles and try again.",
        )
    return ParseResult("ok", "Valid, solvable board.", state)


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
@dataclass
class AppState:
    board: PuzzleState = DEFAULT_BOARD  # last valid board (what the grid shows)
    parse: ParseResult | None = None  # result of parsing the text field
    busy: str | None = None  # None | "solve" | "compare"
    token: CancelToken | None = None
    runs: dict[str, RunResult] = field(
        default_factory=dict
    )  # runs on the current board
    active_key: str | None = None  # run currently shown
    compare: tuple[RunResult, RunResult] | None = None
    step: int = 0
    trace_page: int = 0
    playing: bool = False
    play_gen: int = 0  # invalidates stale playback loops

    @property
    def active(self) -> RunResult | None:
        return self.runs.get(self.active_key) if self.active_key else None

    @property
    def last_step(self) -> int:
        a = self.active
        return len(a.path) - 1 if a else 0


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main(page: ft.Page):
    C = ft.Colors
    s = AppState()

    page.title = "8-Puzzle Solver — Alexandria University Lab 1"
    page.theme_mode = ft.ThemeMode.SYSTEM  # follow the OS light/dark setting
    page.theme = ft.Theme(color_scheme_seed=C.INDIGO)
    page.dark_theme = ft.Theme(color_scheme_seed=C.INDIGO)
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO

    # ---------------------------- small helpers ---------------------------- #
    def card(title: str, body: ft.Control, **kwargs) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[ft.Text(title, size=16, weight=ft.FontWeight.W_600), body],
            ),
            padding=16,
            border_radius=16,
            bgcolor=C.SURFACE_CONTAINER,
            border=ft.Border.all(1, C.OUTLINE_VARIANT),
            **kwargs,
        )

    def muted(text: str = "", size: int = 12, **kwargs) -> ft.Text:
        return ft.Text(text, size=size, color=C.ON_SURFACE_VARIANT, **kwargs)

    def busy_pill(label: str) -> tuple[ft.Container, ft.Text]:
        """Looks like a button, but shows a spinner. Swapped in while a job runs."""
        text = ft.Text(
            label, color=C.ON_SECONDARY_CONTAINER, weight=ft.FontWeight.W_500
        )
        pill = ft.Container(
            visible=False,
            padding=12,
            border_radius=20,
            bgcolor=C.SECONDARY_CONTAINER,
            content=ft.Row(
                tight=True,
                spacing=10,
                controls=[
                    ft.ProgressRing(
                        width=16,
                        height=16,
                        stroke_width=2,
                        color=C.ON_SECONDARY_CONTAINER,
                    ),
                    text,
                ],
            ),
        )
        return pill, text

    # ------------------------------ inputs --------------------------------- #
    state_input = ft.TextField(
        label="Initial state (0 = blank)",
        value=",".join(map(str, DEFAULT_BOARD)),
        col={"xs": 12, "md": 7},
    )
    algo_dropdown = ft.Dropdown(
        label="Search algorithm",
        value="ASTAR_MANHATTAN",
        options=[ft.dropdown.Option(k, v) for k, v in ALGO_OPTIONS.items()],
        col={"xs": 12, "md": 5},
    )
    validation_text = muted(size=13)
    validation_ok_icon = ft.Icon(
        icon=ft.Icons.CHECK_OUTLINED,
        size=16,
        color=C.ON_SURFACE_VARIANT,
        visible=False,
    )
    validation_err_icon = ft.Icon(
        icon=ft.Icons.ERROR_OUTLINE, size=16, color=C.ERROR, visible=False
    )
    validation_row = ft.Row(
        spacing=6,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            validation_ok_icon,
            validation_err_icon,
            ft.Container(content=validation_text, expand=True),
        ],
    )

    solve_btn = ft.FilledButton("Solve", icon=ft.Icons.SEARCH)
    solve_busy, solve_busy_text = busy_pill("Solving…")
    compare_btn = ft.OutlinedButton(
        "Compare A* heuristics", icon=ft.Icons.COMPARE_ARROWS
    )
    compare_busy, compare_busy_text = busy_pill("Comparing…")
    random_btn = ft.OutlinedButton("Random board", icon=ft.Icons.SHUFFLE)
    cancel_btn = ft.OutlinedButton("Cancel", icon=ft.Icons.CLOSE, visible=False)

    # ------------------------------ status --------------------------------- #
    status_spinner = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
    status_text = ft.Text("", expand=True, selectable=True)
    STATUS_STYLES = {  # kind -> (background, text colour, icon)
        "info": (C.PRIMARY_CONTAINER, C.ON_PRIMARY_CONTAINER, ft.Icons.INFO_OUTLINE),
        "busy": (
            C.SECONDARY_CONTAINER,
            C.ON_SECONDARY_CONTAINER,
            None,
        ),  # spinner instead
        "success": (
            C.TERTIARY_CONTAINER,
            C.ON_TERTIARY_CONTAINER,
            ft.Icons.CHECK_OUTLINED,
        ),
        "warn": (
            C.SECONDARY_CONTAINER,
            C.ON_SECONDARY_CONTAINER,
            ft.Icons.WARNING_AMBER_ROUNDED,
        ),
        "error": (C.ERROR_CONTAINER, C.ON_ERROR_CONTAINER, ft.Icons.ERROR_OUTLINE),
    }
    # One Icon per kind, shown/hidden as needed (no need to change an Icon's glyph at runtime).
    status_icons = {
        kind: ft.Icon(icon=icon, size=20, visible=False)
        for kind, (_, _, icon) in STATUS_STYLES.items()
        if icon is not None
    }
    status_box = ft.Container(
        padding=12,
        border_radius=12,
        content=ft.Row(
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[status_spinner, *status_icons.values(), status_text],
        ),
    )

    def set_status(kind: str, message: str) -> None:
        bg, fg, _ = STATUS_STYLES[kind]
        status_box.bgcolor = bg
        status_text.color = fg
        status_text.value = message
        status_spinner.visible = kind == "busy"
        status_spinner.color = fg
        for k, icon in status_icons.items():
            icon.visible = k == kind
            icon.color = fg

    set_status(
        "info",
        "Ready. Enter a board (or pick a random one), choose an algorithm and press Solve.",
    )

    # ------------------------------ board ---------------------------------- #
    GAP = 8
    CELL = TILE + GAP
    BOARD_SIZE = 3 * TILE + 2 * GAP

    # Static slots: the 9 cells of the board (the blank is just an empty slot).
    slot_boxes = [
        ft.Container(
            left=(i % 3) * CELL,
            top=(i // 3) * CELL,
            width=TILE,
            height=TILE,
            border_radius=12,
            bgcolor=C.SURFACE_CONTAINER_HIGHEST,
            border=ft.Border.all(1, C.OUTLINE_VARIANT),
        )
        for i in range(9)
    ]
    # One persistent widget per tile (1..8). Moving a tile = changing its left/top;
    # `animate_position` makes Flet slide it there instead of jumping.
    tile_texts = {
        n: ft.Text(str(n), size=30, weight=ft.FontWeight.BOLD) for n in range(1, 9)
    }
    tile_boxes = {
        n: ft.Container(
            content=tile_texts[n],
            left=0,
            top=0,
            width=TILE,
            height=TILE,
            border_radius=12,
            alignment=ft.Alignment.CENTER,
            animate_position=200,  # ms, updated in refresh()
            animate=150,  # colour changes (in-place highlight)
        )
        for n in range(1, 9)
    }
    board_grid = ft.Stack(
        width=BOARD_SIZE,
        height=BOARD_SIZE,
        controls=cast(list[ft.Control], slot_boxes + list(tile_boxes.values())),
    )

    def legend_item(swatch: ft.Container, text: str) -> ft.Row:
        return ft.Row(tight=True, spacing=6, controls=[swatch, muted(text)])

    def swatch(bg, border=None) -> ft.Container:
        return ft.Container(
            width=14, height=14, border_radius=4, bgcolor=bg, border=border
        )

    legend = ft.Row(
        wrap=True,
        spacing=16,
        run_spacing=4,
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[
            legend_item(swatch(C.PRIMARY_CONTAINER), "Tile"),
            legend_item(
                swatch(C.TERTIARY_CONTAINER, ft.Border.all(2, C.TERTIARY)),
                "In goal position",
            ),
            legend_item(
                swatch(C.SURFACE_CONTAINER_HIGHEST, ft.Border.all(1, C.OUTLINE)),
                "Blank",
            ),
        ],
    )

    step_text = ft.Text(
        "", size=16, weight=ft.FontWeight.W_600, text_align=ft.TextAlign.CENTER
    )
    step_slider = ft.Slider(min=0, max=1, value=0, disabled=True)
    first_btn = ft.IconButton(
        icon=ft.Icons.FIRST_PAGE, tooltip="First step", disabled=True
    )
    prev_btn = ft.IconButton(
        icon=ft.Icons.SKIP_PREVIOUS, tooltip="Previous step", disabled=True
    )
    play_btn = ft.FilledButton(
        "Play",
        icon=ft.Icons.PLAY_ARROW,
        tooltip="Play (restarts at the end)",
        disabled=True,
    )
    pause_btn = ft.FilledButton("Pause", icon=ft.Icons.PAUSE, visible=False)
    next_btn = ft.IconButton(
        icon=ft.Icons.SKIP_NEXT, tooltip="Next step", disabled=True
    )
    last_btn = ft.IconButton(
        icon=ft.Icons.LAST_PAGE, tooltip="Last step", disabled=True
    )
    speed_dropdown = ft.Dropdown(
        label="Playback speed",
        value="normal",
        width=170,
        options=[
            ft.dropdown.Option("slow", "Slow"),
            ft.dropdown.Option("normal", "Normal"),
            ft.dropdown.Option("fast", "Fast"),
            ft.dropdown.Option("turbo", "Turbo"),
        ],
    )

    # ----------------------------- results --------------------------------- #
    def make_metric(title: str) -> tuple[ft.Container, ft.Text]:
        value = ft.Text(
            "-",
            size=20,
            weight=ft.FontWeight.BOLD,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        box = ft.Container(
            col={"xs": 6, "sm": 3},
            padding=12,
            border_radius=12,
            bgcolor=C.SURFACE_CONTAINER_HIGH,
            content=ft.Column(spacing=2, tight=True, controls=[muted(title), value]),
        )
        return box, value

    m_cost_box, m_cost = make_metric("Cost of path")
    m_exp_box, m_exp = make_metric("Nodes expanded")
    m_depth_box, m_depth = make_metric("Depth (goal / max)")
    m_time_box, m_time = make_metric("Running time")
    metrics_row = ft.ResponsiveRow(
        spacing=12,
        run_spacing=12,
        controls=[m_cost_box, m_exp_box, m_depth_box, m_time_box],
    )
    result_title = ft.Text("", size=13, color=C.ON_SURFACE_VARIANT)

    path_text = ft.Text("", selectable=True, size=14)
    path_note = muted()
    save_btn = ft.OutlinedButton(
        "Save full trace (.txt)", icon=ft.Icons.DOWNLOAD, disabled=True
    )
    file_picker = ft.FilePicker()
    page.services.append(file_picker)
    path_body = ft.Column(
        spacing=10, controls=[path_text, path_note, ft.Row([save_btn])]
    )

    trace_range = muted(size=13)
    trace_first = ft.IconButton(
        icon=ft.Icons.FIRST_PAGE, tooltip="First page", icon_size=20
    )
    trace_prev = ft.IconButton(
        icon=ft.Icons.CHEVRON_LEFT, tooltip="Previous page", icon_size=20
    )
    trace_next = ft.IconButton(
        icon=ft.Icons.CHEVRON_RIGHT, tooltip="Next page", icon_size=20
    )
    trace_last = ft.IconButton(
        icon=ft.Icons.LAST_PAGE, tooltip="Last page", icon_size=20
    )
    trace_rows = ft.Column(spacing=4)
    trace_body = ft.Column(
        spacing=8,
        controls=[
            ft.Row(
                wrap=True,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    trace_range,
                    ft.Row(
                        tight=True,
                        spacing=0,
                        controls=[trace_first, trace_prev, trace_next, trace_last],
                    ),
                ],
            ),
            trace_rows,
            muted("Tap a step to jump to it on the board."),
        ],
    )

    compare_body = ft.Column(spacing=8)
    history_body = ft.Column(spacing=4)

    path_card = card("Path to goal", path_body)
    trace_card = card("Step-by-step trace", trace_body)
    compare_card = card("A* heuristic comparison", compare_body, visible=False)
    history_card = card("Runs on this board", history_body, visible=False)

    results_col = ft.Column(
        spacing=16,
        visible=False,
        controls=[
            ft.Column(spacing=8, controls=[result_title, metrics_row]),
            path_card,
            trace_card,
            compare_card,
            history_card,
        ],
    )
    empty_card = ft.Container(
        padding=24,
        alignment=ft.Alignment.CENTER,
        border_radius=16,
        border=ft.Border.all(1, C.OUTLINE_VARIANT),
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.INSIGHTS, size=36, color=C.ON_SURFACE_VARIANT),
                ft.Text("No results yet", size=16, weight=ft.FontWeight.W_600),
                muted(
                    "Press Solve to see the path, cost, nodes expanded, depth and running time."
                ),
            ],
        ),
    )

    # --------------------------- table helpers ----------------------------- #
    def table_row(
        cells: list[str], weights: list[int], header=False, active=False, on_click=None
    ) -> ft.Container:
        texts = []
        for i, (c, w) in enumerate(zip(cells, weights, strict=True)):
            texts.append(
                ft.Text(
                    c,
                    expand=w,
                    size=12 if header else 13,
                    weight=ft.FontWeight.W_600 if (header or active) else None,
                    color=C.ON_SURFACE_VARIANT if header else None,
                    text_align=ft.TextAlign.LEFT if i == 0 else ft.TextAlign.RIGHT,
                )
            )
        return ft.Container(
            content=ft.Row(controls=texts, spacing=8),
            padding=8,
            border_radius=8,
            bgcolor=C.PRIMARY_CONTAINER if active else None,
            on_click=on_click,
            ink=on_click is not None,
        )

    # -------------------------- rendering functions ------------------------ #
    def render_results() -> None:
        """Rebuilds everything that depends on WHICH run is shown (not the step)."""
        r = s.active
        has = r is not None and r.solved
        results_col.visible = has
        empty_card.visible = not has
        if not has or r is None:
            compare_card.visible = False
            history_card.visible = False
            return

        result_title.value = f"Showing: {r.label}"
        m_cost.value = f"{int(r.cost)}"
        m_exp.value = f"{r.expanded:,}"
        m_depth.value = f"{r.goal_depth} / {r.max_depth}"
        m_time.value = format_time(r.time_ms)

        moves = r.moves
        if not moves:
            path_text.value = "Already at the goal - no moves needed."
        elif len(moves) <= 30:
            path_text.value = " → ".join(moves)
        else:
            shown = " ".join(ARROWS[m] for m in moves[:MAX_PATH_PREVIEW])
            path_text.value = shown + (" …" if len(moves) > MAX_PATH_PREVIEW else "")
        if len(moves) > MAX_PATH_PREVIEW:
            path_note.value = (
                f"{len(moves):,} moves in total - first {MAX_PATH_PREVIEW} shown. "
                "Use “Save full trace” for every step. ↑ ↓ ← → = direction the blank moves."
            )
        elif len(moves) > 30:
            path_note.value = (
                f"{len(moves)} moves. ↑ ↓ ← → = direction the blank moves."
            )
        else:
            path_note.value = (
                f"{len(moves)} move(s). Each move is the direction the blank slides."
            )

        # Comparison card (only after "Compare A* heuristics")
        compare_card.visible = s.compare is not None
        if s.compare is not None:
            compare_body.controls = build_comparison(*s.compare)

        # History card (only useful once there are 2+ runs)
        history_card.visible = len(s.runs) >= 2
        weights = [4, 2, 3, 3, 3]
        rows: list[ft.Control] = [
            table_row(
                ["Algorithm", "Cost", "Expanded", "Max depth", "Time"],
                weights,
                header=True,
            )
        ]
        for k, run in s.runs.items():
            rows.append(
                table_row(
                    [
                        run.label,
                        f"{int(run.cost)}",
                        f"{run.expanded:,}",
                        f"{run.max_depth:,}",
                        format_time(run.time_ms),
                    ],
                    weights,
                    active=k == s.active_key,
                    on_click=lambda _, key=k: select_run(key),
                )
            )
        rows.append(muted("Tap a row to view that run's path and trace."))
        history_body.controls = rows

    def build_comparison(m: RunResult, e: RunResult) -> list[ft.Control]:
        w = [4, 3, 3]
        rows: list[ft.Control] = [
            table_row(["", "Manhattan", "Euclidean"], w, header=True),
            table_row(["Nodes expanded", f"{m.expanded:,}", f"{e.expanded:,}"], w),
            table_row(["Path cost", f"{int(m.cost)}", f"{int(e.cost)}"], w),
            table_row(
                [
                    "Depth (goal / max)",
                    f"{m.goal_depth} / {m.max_depth}",
                    f"{e.goal_depth} / {e.max_depth}",
                ],
                w,
            ),
            table_row(
                ["Running time", format_time(m.time_ms), format_time(e.time_ms)], w
            ),
        ]
        notes: list[str] = []
        if m.expanded < e.expanded:
            pct = 100 * (e.expanded - m.expanded) / e.expanded
            notes.append(
                f"Manhattan expanded {e.expanded - m.expanded:,} fewer nodes ({pct:.0f}% fewer) than Euclidean."
            )
        elif m.expanded > e.expanded:
            notes.append(
                "Euclidean expanded fewer nodes on this board (a tie-breaking effect)."
            )
        else:
            notes.append("Both expanded the same number of nodes on this board.")
        if m.moves == e.moves:
            notes.append(f"Output paths are identical ({len(m.moves)} moves).")
        elif m.cost == e.cost:
            notes.append(
                f"Output paths differ but both are optimal ({int(m.cost)} moves)."
            )
        else:
            notes.append(
                f"Output paths differ in cost ({int(m.cost)} vs {int(e.cost)})."
            )
        notes.append(
            "Both heuristics are admissible (never overestimate the remaining moves), so A* is optimal "
            "with either. Manhattan is the tighter, more informed bound (h_Manhattan ≥ h_Euclidean for "
            "every state), which is why it generally expands fewer nodes."
        )
        rows.append(ft.Divider(height=8))
        rows.extend(ft.Text(n, size=13) for n in notes)
        return rows

    def render_trace() -> None:
        r = s.active
        if r is None:
            trace_rows.controls = []
            return
        total = len(r.path)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        s.trace_page = max(0, min(s.trace_page, pages - 1))
        start = s.trace_page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        trace_range.value = f"Steps {start}-{end - 1} of {total - 1}"
        trace_first.disabled = trace_prev.disabled = s.trace_page == 0
        trace_next.disabled = trace_last.disabled = s.trace_page >= pages - 1

        rows = []
        for i in range(start, end):
            action, st = r.path[i]
            is_current = i == s.step
            move = "Start" if action is None else f"{ARROWS[action]} {action}"
            rows.append(
                ft.Container(
                    padding=8,
                    border_radius=8,
                    ink=True,
                    bgcolor=C.PRIMARY_CONTAINER
                    if is_current
                    else C.SURFACE_CONTAINER_HIGH,
                    on_click=lambda _, idx=i: goto(idx),
                    content=ft.Row(
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                        controls=[
                            ft.Text(
                                f"#{i}",
                                width=58,
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                font_family="monospace",
                            ),
                            ft.Text(move, width=72, size=13),
                            ft.Text(
                                format_board_compact(st),
                                expand=True,
                                size=13,
                                font_family="monospace",
                            ),
                        ],
                    ),
                )
            )
        trace_rows.controls = rows

    def animation_ms() -> int:
        """Slide duration: shorter than the playback delay so tiles finish before the next move."""
        if s.playing:
            delay_ms = SPEEDS.get(speed_dropdown.value or "normal", 0.45) * 1000
            return max(1, min(250, int(delay_ms * 0.8)))
        return 200

    def refresh(skip_slider: bool = False, instant: bool = False) -> None:
        r = s.active
        has = r is not None and r.solved
        busy = s.busy is not None
        last = s.last_step if has else 0

        # Board: slide each tile to its cell
        view = r.path[s.step][1] if (has and r is not None) else s.board
        ms = 1 if instant else animation_ms()
        for n in range(1, 9):
            pos = view.index(n)
            box, txt = tile_boxes[n], tile_texts[n]
            box.animate_position = ms
            box.left = (pos % 3) * CELL
            box.top = (pos // 3) * CELL
            if n == pos:  # tile is in its goal cell
                txt.color = C.ON_TERTIARY_CONTAINER
                box.bgcolor = C.TERTIARY_CONTAINER
                box.border = ft.Border.all(2, C.TERTIARY)
            else:
                txt.color = C.ON_PRIMARY_CONTAINER
                box.bgcolor = C.PRIMARY_CONTAINER
                box.border = None

        # Step caption
        if has and r is not None:
            action = r.path[s.step][0]
            what = (
                "Initial state" if action is None else f"{ARROWS[action]} Move {action}"
            )
            tail = "  ·  ✓ Goal reached" if s.step == last else ""
            step_text.value = f"Step {s.step} / {last}  ·  {what}{tail}"
        else:
            step_text.value = "Preview of the initial board"

        # Player controls
        locked = (not has) or busy or last == 0
        step_slider.max = max(1, last)
        step_slider.divisions = last if 0 < last <= 60 else None
        if not skip_slider:
            step_slider.value = s.step
        step_slider.disabled = locked or s.playing
        first_btn.disabled = prev_btn.disabled = locked or s.playing or s.step == 0
        next_btn.disabled = last_btn.disabled = locked or s.playing or s.step >= last
        play_btn.visible = not s.playing
        play_btn.disabled = locked
        pause_btn.visible = s.playing
        speed_dropdown.disabled = busy
        save_btn.disabled = (not has) or busy

        # Setup controls
        valid = s.parse is not None and s.parse.state is not None
        can_run = valid and not busy
        solve_btn.visible = s.busy != "solve"
        solve_busy.visible = s.busy == "solve"
        compare_btn.visible = s.busy != "compare"
        compare_busy.visible = s.busy == "compare"
        solve_btn.disabled = compare_btn.disabled = not can_run
        random_btn.disabled = busy
        cancel_btn.visible = busy
        state_input.disabled = busy
        algo_dropdown.disabled = busy

        render_trace()
        page.update()

    # ------------------------------ handlers ------------------------------- #
    def stop_playback() -> None:
        s.playing = False
        s.play_gen += 1

    def reset_results() -> None:
        stop_playback()
        s.runs.clear()
        s.active_key = None
        s.compare = None
        s.step = 0
        s.trace_page = 0
        render_results()

    def apply_input_validation() -> None:
        res = parse_board(state_input.value or "")
        s.parse = res
        validation_text.value = res.message
        validation_ok_icon.visible = res.level == "ok"
        validation_err_icon.visible = res.level == "error"
        validation_text.color = (
            C.ERROR if res.level == "error" else C.ON_SURFACE_VARIANT
        )
        state_input.border_color = C.ERROR if res.level == "error" else None
        if res.state is not None and res.state != s.board:
            s.board = res.state
            reset_results()  # results belong to the previous board
            set_status("info", "Board changed. Press Solve to search from this board.")

    def on_input_change(e) -> None:
        apply_input_validation()
        refresh(instant=True)

    def on_random(e) -> None:
        if s.busy:
            return
        new = s.board
        while new == s.board or new == tuple(range(9)):
            new = EightPuzzleProblem.random_solvable_state(steps=20)
        state_input.value = ",".join(map(str, new))
        apply_input_validation()
        set_status("info", "Random solvable board generated. Press Solve to search.")
        refresh(instant=True)

    async def run_job(kind: str, keys: list[str]) -> None:
        """Runs one or more searches off the UI thread while showing progress."""
        if s.busy or s.parse is None or s.parse.state is None:
            return
        board = s.parse.state
        token = CancelToken()
        s.busy, s.token = kind, token
        stop_playback()
        names = " and ".join(ALGORITHMS[k][0] for k in keys)
        (solve_busy_text if kind == "solve" else compare_busy_text).value = (
            "Solving…" if kind == "solve" else "Comparing…"
        )
        set_status(
            "busy",
            f"Running {names}… large boards can take a while with DFS / Iterative DFS.",
        )
        refresh()

        try:
            results = await asyncio.to_thread(
                lambda: [execute_search(k, board, token) for k in keys]
            )
        except SearchCancelled:
            set_status("warn", "Search cancelled.")
        except Exception as ex:
            set_status("error", f"Search failed: {ex}")
        else:
            if not all(r.solved for r in results):
                set_status("error", "No solution found for this board.")
            else:
                for r in results:
                    s.runs[r.key] = r
                s.active_key = results[0].key
                s.step = 0
                s.trace_page = 0
                s.compare = (results[0], results[1]) if kind == "compare" else None
                r0 = results[0]
                if kind == "compare":
                    m, e = results
                    set_status(
                        "success",
                        f"Compared A* heuristics: Manhattan expanded {m.expanded:,} nodes, "
                        f"Euclidean {e.expanded:,}. Both cost {int(m.cost)}.",
                    )
                else:
                    extra = (
                        " DFS does not guarantee the shortest path."
                        if r0.key == "DFS"
                        else ""
                    )
                    set_status(
                        "success",
                        f"Solved with {r0.label}: {int(r0.cost)} moves, {r0.expanded:,} nodes expanded "
                        f"in {format_time(r0.time_ms)}.{extra}",
                    )
                if len(r0.path) <= MAX_CONSOLE_TRACE:
                    print_trace(r0)
        finally:
            s.busy, s.token = None, None
        render_results()
        refresh(instant=True)

    def print_trace(r: RunResult) -> None:
        """Console trace of every step (kept for short paths; long ones go to the .txt export)."""
        print(f"\n=== {r.label} | cost {int(r.cost)} | expanded {r.expanded} ===")
        for i, (a, st) in enumerate(r.path):
            print(
                f"Step {i}: {'Initial state' if a is None else f'Move {a}'}\n{format_board_ascii(st)}\n"
            )

    async def on_solve(e) -> None:
        await run_job("solve", [algo_dropdown.value or "ASTAR_MANHATTAN"])

    async def on_compare(e) -> None:
        await run_job("compare", ["ASTAR_MANHATTAN", "ASTAR_EUCLIDEAN"])

    def on_cancel(e) -> None:
        if s.token is not None:
            s.token.cancel()
            cancel_btn.disabled = True
            set_status("busy", "Cancelling…")
            page.update()

    def select_run(key: str) -> None:
        if s.busy or key not in s.runs:
            return
        stop_playback()
        s.active_key = key
        s.step = 0
        s.trace_page = 0
        render_results()
        refresh(instant=True)

    def goto(idx: int, from_slider: bool = False) -> None:
        if s.active is None or s.busy:
            return
        s.step = max(0, min(int(idx), s.last_step))
        s.trace_page = s.step // PAGE_SIZE  # trace follows the current step
        refresh(skip_slider=from_slider, instant=from_slider)

    async def play_loop(gen: int) -> None:
        while s.playing and gen == s.play_gen:
            await asyncio.sleep(SPEEDS.get(speed_dropdown.value or "normal", 0.45))
            if not s.playing or gen != s.play_gen:
                return
            if s.step >= s.last_step:
                break
            goto(s.step + 1)
        if gen == s.play_gen:
            s.playing = False
            refresh()

    def on_play(e) -> None:
        if s.active is None or s.busy or s.last_step == 0:
            return
        if s.step >= s.last_step:
            s.step, s.trace_page = 0, 0
        s.playing = True
        s.play_gen += 1
        refresh(instant=s.step == 0)
        page.run_task(play_loop, s.play_gen)

    def on_pause(e) -> None:
        stop_playback()
        refresh()

    async def on_save_trace(e) -> None:
        r = s.active
        if r is None or s.busy:
            return
        board = s.board
        save_btn.disabled = True
        set_status("busy", f"Preparing {len(r.path):,} steps…")
        page.update()
        try:
            # Build the bytes off the UI thread (DFS traces are ~7 MB)
            data = await asyncio.to_thread(
                lambda: build_trace_text(r, board).encode("utf-8")
            )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = await file_picker.save_file(
                dialog_title="Save trace",
                file_name=f"trace_{r.key.lower()}_{stamp}.txt",
                src_bytes=data,  # required on Android / iOS / web
            )
            is_mobile = page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS)
            if page.web or is_mobile:
                # The system dialog saves the bytes itself and may not report a path back.
                set_status(
                    "info",
                    "Save dialog closed. If you picked a location, the trace was saved there.",
                )
            elif path:
                # Desktop: write to the location the user picked (harmless if already written).
                await asyncio.to_thread(Path(path).write_bytes, data)
                set_status("success", f"Trace saved to {path}")
            else:
                set_status("warn", "Save cancelled.")
        except Exception as ex:
            set_status("error", f"Could not save the trace: {ex}")
        refresh()

    def trace_page_to(page_idx: int) -> None:
        s.trace_page = page_idx  # clamped to the valid range in render_trace()
        render_trace()
        page.update()

    # wire events
    state_input.on_change = on_input_change
    state_input.on_submit = on_solve
    solve_btn.on_click = on_solve
    compare_btn.on_click = on_compare
    random_btn.on_click = on_random
    cancel_btn.on_click = on_cancel
    first_btn.on_click = lambda e: goto(0)
    prev_btn.on_click = lambda e: goto(s.step - 1)
    next_btn.on_click = lambda e: goto(s.step + 1)
    last_btn.on_click = lambda e: goto(s.last_step)
    play_btn.on_click = on_play
    pause_btn.on_click = on_pause
    save_btn.on_click = on_save_trace
    step_slider.on_change = lambda e: goto(
        round(cast(ft.Number, e.control.value)), from_slider=True
    )
    trace_first.on_click = lambda e: trace_page_to(0)
    trace_prev.on_click = lambda e: trace_page_to(s.trace_page - 1)
    trace_next.on_click = lambda e: trace_page_to(s.trace_page + 1)
    trace_last.on_click = lambda e: trace_page_to(10**9)

    # ------------------------------ layout --------------------------------- #
    setup_card = card(
        "Setup",
        ft.Column(
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                ft.ResponsiveRow(
                    spacing=12, run_spacing=12, controls=[state_input, algo_dropdown]
                ),
                validation_row,
                ft.Row(
                    wrap=True,
                    spacing=10,
                    run_spacing=10,
                    controls=[
                        solve_btn,
                        solve_busy,
                        compare_btn,
                        compare_busy,
                        random_btn,
                        cancel_btn,
                    ],
                ),
            ],
        ),
    )

    board_card = card(
        "Board",
        ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=14,
            controls=[
                board_grid,
                legend,
                step_text,
                ft.Container(content=step_slider, width=TILE * 3 + 16),
                ft.Row(
                    wrap=True,
                    spacing=4,
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[
                        first_btn,
                        prev_btn,
                        play_btn,
                        pause_btn,
                        next_btn,
                        last_btn,
                    ],
                ),
                speed_dropdown,
            ],
        ),
    )

    page.add(
        ft.SafeArea(
            content=ft.Column(
                spacing=16,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Column(
                        spacing=2,
                        controls=[
                            ft.Text(
                                "8-Puzzle Solver", size=28, weight=ft.FontWeight.BOLD
                            ),
                            muted(
                                "Informed & uninformed search: BFS · DFS · Iterative DFS · A*",
                                size=14,
                            ),
                        ],
                    ),
                    setup_card,
                    status_box,
                    ft.ResponsiveRow(
                        spacing=16,
                        run_spacing=16,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        controls=[
                            ft.Container(content=board_card, col={"xs": 12, "lg": 5}),
                            ft.Container(
                                content=ft.Column(
                                    spacing=16, controls=[empty_card, results_col]
                                ),
                                col={"xs": 12, "lg": 7},
                            ),
                        ],
                    ),
                ],
            )
        )
    )

    apply_input_validation()
    render_results()
    refresh(instant=True)


if __name__ == "__main__":
    ft.run(main)
