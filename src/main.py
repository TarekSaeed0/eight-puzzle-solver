import asyncio
import flet as ft
from search.algorithm import (
    AStarSearchAlgorithm,
    BreadthFirstSearchAlgorithm,
    DepthFirstSearchAlgorithm,
    IterativeDeepeningDepthFirstSearchAlgorithm,
    ObservableSearchAlgorithm,
)
from search.metrics import PuzzleMetricsObserver
from search.puzzle import EightPuzzleProblem, PuzzleAction, PuzzleState


def format_board_ascii(state: PuzzleState) -> str:
    """Formats a 9-tuple into a clean 3x3 text grid for traceable output."""
    lines = []
    for r in range(3):
        row_vals = [str(state[r * 3 + c]) if state[r * 3 + c] != 0 else " " for c in range(3)]
        lines.append(" | ".join(row_vals))
    return "\n---------\n".join(lines)


def main(page: ft.Page):
    page.title = "8-Puzzle AI Solver — Alexandria University Lab 1"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 24
    page.scroll = ft.ScrollMode.AUTO

    # Application State
    current_initial_state: PuzzleState = (1, 2, 5, 3, 4, 0, 6, 7, 8)
    solution_path: list[tuple[PuzzleAction | None, PuzzleState]] = [
        (None, current_initial_state)
    ]
    current_step_idx: int = 0
    is_playing: bool = False


    state_input = ft.TextField(
        label="Initial State (comma-separated 0-8)",
        value="1,2,5,3,4,0,6,7,8",
        width=280,
    )

    algo_dropdown = ft.Dropdown(
        label="Search Algorithm",
        width=280,
        value="ASTAR_MANHATTAN",
        options=[
            ft.dropdown.Option("BFS", "BFS (Breadth-First Search)"),
            ft.dropdown.Option("DFS", "DFS (Depth-First Search)"),
            ft.dropdown.Option("IDDFS", "Iterative Deepening DFS (IDDFS)"),
            ft.dropdown.Option("ASTAR_MANHATTAN", "A* (Manhattan Heuristic)"),
            ft.dropdown.Option("ASTAR_EUCLIDEAN", "A* (Euclidean Heuristic)"),
        ],
    )

    status_banner = ft.Text(
        "Ready. Select an algorithm and click Solve.",
        size=14,
        color=ft.Colors.BLUE_200,
    )

    # --- UI Controls: 3x3 Puzzle Grid ---
    tile_containers: list[ft.Container] = []
    tile_texts: list[ft.Text] = []

    for _ in range(9):
        t_text = ft.Text("", size=32, weight=ft.FontWeight.BOLD)
        t_box = ft.Container(
            content=t_text,
            width=88,
            height=88,
            border_radius=12,
            alignment=ft.Alignment.CENTER,
        )
        tile_texts.append(t_text)
        tile_containers.append(t_box)

    board_ui = ft.Column(
        spacing=8,
        controls=[
            ft.Row(spacing=8, controls=tile_containers[0:3], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(spacing=8, controls=tile_containers[3:6], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row(spacing=8, controls=tile_containers[6:9], alignment=ft.MainAxisAlignment.CENTER),
        ],
    )

    step_indicator = ft.Text(
        "Step: 0 / 0 | Action: Initial",
        size=16,
        weight=ft.FontWeight.W_600,
    )

    # --- UI Controls: Deliverables Metrics ---
    metric_cost = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)
    metric_expanded = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)
    metric_depth = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)
    metric_time = ft.Text("-", size=20, weight=ft.FontWeight.BOLD)

    def create_metric_card(title: str, value_control: ft.Text) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(title, size=12, color=ft.Colors.GREY_400),
                    value_control,
                ],
                spacing=4,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=14,
            width=185,
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=10,
        )

    metrics_row = ft.Row(
        wrap=True,
        spacing=12,
        controls=[
            create_metric_card("Cost of Path", metric_cost),
            create_metric_card("Nodes Expanded", metric_expanded),
            create_metric_card("Search Depth (Goal / Max)", metric_depth),
            create_metric_card("Running Time", metric_time),
        ],
    )

    trace_list = ft.ListView(
        expand=True,
        spacing=8,
        height=320,
    )

    def render_board_state(idx: int):
        nonlocal current_step_idx
        if not solution_path:
            return
        current_step_idx = max(0, min(idx, len(solution_path) - 1))
        action, state = solution_path[current_step_idx]

        for i, val in enumerate(state):
            if val == 0:
                tile_texts[i].value = ""
                tile_containers[i].bgcolor = ft.Colors.GREY_900
                tile_containers[i].border = ft.Border.all(2, ft.Colors.GREY_800)
            else:
                tile_texts[i].value = str(val)
                # Highlight tiles that are in their exact goal position (val == i)
                is_in_place = val == i
                tile_containers[i].bgcolor = (
                    ft.Colors.TEAL_700 if is_in_place else ft.Colors.INDIGO_600
                )
                tile_containers[i].border = None

        action_str = action if action else "Initial State"
        step_indicator.value = (
            f"Step: {current_step_idx} / {len(solution_path) - 1}  |  Action: {action_str}"
        )
        prev_btn.disabled = current_step_idx == 0
        next_btn.disabled = current_step_idx >= len(solution_path) - 1
        page.update()

    def parse_input_state() -> PuzzleState | None:
        try:
            parts = [int(x.strip()) for x in state_input.value.split(",")]
            state = tuple(parts)
            if len(state) != 9 or set(state) != set(range(9)):
                status_banner.value = "Error: Enter all digits from 0 to 8 exactly once."
                status_banner.color = ft.Colors.RED_400
                page.update()
                return None
            if not EightPuzzleProblem.is_solvable(state):
                status_banner.value = (
                    "Unsolvable State: Odd number of inversions! Try another configuration."
                )
                status_banner.color = ft.Colors.AMBER_400
                page.update()
                return None
            return state
        except ValueError:
            status_banner.value = "Error: Invalid format. Use comma-separated integers (e.g., 1,2,5,3,4,0,6,7,8)."
            status_banner.color = ft.Colors.RED_400
            page.update()
            return None

    def on_randomize_click(e):
        nonlocal is_playing, current_initial_state, solution_path
        is_playing = False
        rand_state = EightPuzzleProblem.random_solvable_state(steps=20)
        current_initial_state = rand_state
        state_input.value = ",".join(map(str, rand_state))
        solution_path = [(None, rand_state)]
        trace_list.controls.clear()
        status_banner.value = "Generated a random solvable 8-puzzle state."
        status_banner.color = ft.Colors.GREEN_300
        render_board_state(0)

    def build_algorithm(key: str) -> tuple[ObservableSearchAlgorithm, str]:
        if key == "BFS":
            return BreadthFirstSearchAlgorithm(), "none"
        if key == "DFS":
            return DepthFirstSearchAlgorithm(), "none"
        if key == "IDDFS":
            return IterativeDeepeningDepthFirstSearchAlgorithm(), "none"
        if key == "ASTAR_MANHATTAN":
            return AStarSearchAlgorithm(), "manhattan"
        if key == "ASTAR_EUCLIDEAN":
            return AStarSearchAlgorithm(), "euclidean"
        return AStarSearchAlgorithm(), "manhattan"

    def on_solve_click(e):
        nonlocal is_playing, solution_path, current_initial_state
        is_playing = False
        parsed = parse_input_state()
        if parsed is None:
            return

        current_initial_state = parsed
        algo_key = algo_dropdown.value or "ASTAR_MANHATTAN"
        algorithm, h_type = build_algorithm(algo_key)

        problem = EightPuzzleProblem(
            initial=current_initial_state,
            heuristic_type=h_type,  
        )
        observer = PuzzleMetricsObserver()
        algorithm.add_observer(observer)

        status_banner.value = f"Running {algo_key}..."
        status_banner.color = ft.Colors.BLUE_200
        page.update()

        # Execute search
        algorithm.search(problem)

        if observer.solved:
            solution_path = observer.path
            actions_seq = [a for a, _ in solution_path if a is not None]

            metric_cost.value = f"{int(observer.path_cost)}"
            metric_expanded.value = f"{observer.nodes_expanded:,}"
            metric_depth.value = f"{observer.goal_depth} / {observer.max_search_depth}"
            metric_time.value = f"{observer.running_time_ms:.2f} ms"

            trace_list.controls.clear()
            for step_i, (act, st) in enumerate(solution_path):
                header = f"Step {step_i}: {'Initial State' if act is None else f'Move {act}'}"
                trace_list.controls.append(
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Text(header, width=140, weight=ft.FontWeight.BOLD),
                                ft.Text(
                                    format_board_ascii(st),
                                    font_family="monospace",
                                    size=13,
                                ),
                                ft.Text(f"Tuple: {st}", size=12, color=ft.Colors.GREY_400),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        padding=10,
                        bgcolor=ft.Colors.GREY_900,
                        border_radius=8,
                        on_click=lambda _, idx=step_i: render_board_state(idx),
                    )
                )

            preview_moves = " -> ".join(actions_seq[:12])
            if len(actions_seq) > 12:
                preview_moves += f" ... (+{len(actions_seq) - 12} more)"
            status_banner.value = f"Solved! Path to Goal: [{preview_moves or 'None'}]"
            status_banner.color = ft.Colors.GREEN_400
            render_board_state(0)
        else:
            status_banner.value = "No solution found."
            status_banner.color = ft.Colors.RED_400
            page.update()

    def on_compare_heuristics_click(e):
        """Directly compares Manhattan vs Euclidean A* on the current board for Section 2.2."""
        parsed = parse_input_state()
        if parsed is None:
            return

        results = {}
        for h_name in ("manhattan", "euclidean"):
            prob = EightPuzzleProblem(initial=parsed, heuristic_type=h_name)  # type: ignore
            algo = AStarSearchAlgorithm()
            obs = PuzzleMetricsObserver()
            algo.add_observer(obs)
            algo.search(prob)
            results[h_name] = obs

        m_obs = results["manhattan"]
        e_obs = results["euclidean"]

        status_banner.value = (
            f"A* Comparison | Manhattan: {m_obs.nodes_expanded} nodes expanded, "
            f"Cost={int(m_obs.path_cost)}, Time={m_obs.running_time_ms:.2f}ms  ||  "
            f"Euclidean: {e_obs.nodes_expanded} nodes expanded, "
            f"Cost={int(e_obs.path_cost)}, Time={e_obs.running_time_ms:.2f}ms"
        )
        status_banner.color = ft.Colors.CYAN_300
        page.update()

    async def animate_playback():
        nonlocal is_playing
        while is_playing and current_step_idx < len(solution_path) - 1:
            render_board_state(current_step_idx + 1)
            await asyncio.sleep(0.45)
        is_playing = False
        play_btn.text = "Play Solution"
        play_btn.icon = "play_arrow"
        page.update()

    def on_play_pause_click(e):
        nonlocal is_playing
        if len(solution_path) <= 1:
            return
        if is_playing:
            is_playing = False
            play_btn.text = "Play Solution"
            play_btn.icon = "play_arrow"
            page.update()
        else:
            if current_step_idx >= len(solution_path) - 1:
                render_board_state(0)
            is_playing = True
            play_btn.text = "Pause"
            play_btn.icon = "pause"
            page.update()
            page.run_task(animate_playback)
    prev_btn = ft.TextButton(
        "Prev",
        icon="skip_previous",
        on_click=lambda e: render_board_state(current_step_idx - 1),
    )
    play_btn = ft.TextButton(
        "Play Solution",
        icon="play_arrow",
        on_click=on_play_pause_click,
    )
    next_btn = ft.TextButton(
        "Next",
        icon="skip_next",
        on_click=lambda e: render_board_state(current_step_idx + 1),
    )

    # --- Layout Assembly ---
    page.add(
        ft.Column(
            spacing=18,
            controls=[
                ft.Text(
                    "8-Puzzle Solver (Informed & Uninformed Search)",
                    size=26,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Row(
                    wrap=True,
                    spacing=12,
                    controls=[
                        state_input,
                        algo_dropdown,
                        ft.TextButton("Solve", icon="search", on_click=on_solve_click),
                        ft.TextButton("Random Board", icon="shuffle", on_click=on_randomize_click),
                        ft.TextButton(
                            "Compare A* Heuristics",
                            icon="compare_arrows",
                            on_click=on_compare_heuristics_click,
                        ),
                    ],
                ),
                status_banner,
                metrics_row,
                ft.Divider(),
                ft.Row(
                    wrap=True,
                    spacing=36,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        ft.Column(
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=14,
                            controls=[
                                board_ui,
                                step_indicator,
                                ft.Row([prev_btn, play_btn, next_btn], spacing=10),
                            ],
                        ),
                        ft.Container(
                            expand=True,
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "Traceable Step-by-Step Path (Click any step to jump):",
                                        size=16,
                                        weight=ft.FontWeight.W_600,
                                    ),
                                    trace_list,
                                ]
                            ),
                        ),
                    ],
                ),
            ],
        )
    )

    render_board_state(0)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("Starting Flet 8-Puzzle...")
    
    try:
        ft.run(main, view=ft.AppView.WEB_BROWSER)
    except Exception as e:
        print(f"Failed to launch Flet: {e}")