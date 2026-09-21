"""Dependency-free, cell-aware arcade presentation of an authoritative snapshot."""

from __future__ import annotations

import math
import re
import unicodedata

BG = "091318"
PANEL = "0d1d23"
GRID = "193138"
BORDER = "28515a"
TEXT = "e4f7f3"
MUTED = "829da5"
GREEN = "77f2be"
CYAN = "79d9ed"
AMBER = "ffc47a"
RED = "ff8594"

DIGITS = {
    "0": ("█▀█", "█ █", "▀▀▀"), "1": ("▄█ ", " █ ", "▀▀▀"),
    "2": ("▀▀█", "█▀▀", "▀▀▀"), "3": ("▀▀█", "▀▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"), "5": ("█▀▀", "▀▀█", "▀▀▀"),
    "6": ("█▀▀", "█▀█", "▀▀▀"), "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"), "9": ("█▀█", "▀▀█", "▀▀▀"),
}
ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|.)")


def clean_text(value):
    """Never let model/server text inject terminal commands or change a row."""
    text = _ESCAPE.sub("", str(value))
    return "".join(" " if char in "\r\n\t" else char for char in text
                   if char in "\r\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"})


def cell_width(text):
    return sum(0 if unicodedata.combining(char) or unicodedata.category(char) in {"Mn", "Me"}
               else 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
               for char in clean_text(text))


def _rgb(value):
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _mix(start, end, fraction):
    return "".join(f"{round(a + (b - a) * fraction):02x}"
                   for a, b in zip(_rgb(start), _rgb(end)))


class Canvas:
    """One slot per terminal cell, including explicit wide-character continuations."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.styles = [[(TEXT, BG)] * width for _ in range(height)]

    def put(self, row, column, value, color=TEXT, background=None, *, limit=None):
        if not 0 <= row < self.height:
            return
        end = min(self.width, column + limit) if limit is not None else self.width
        previous = None
        for char in clean_text(value):
            size = cell_width(char)
            if size == 0:
                if previous is not None:
                    self.chars[row][previous] += char
                continue
            if column + size > end:
                break
            if column >= 0:
                # Clear a wide glyph if a later drawing overlaps either half.
                for cell in range(column, column + size):
                    if self.chars[row][cell] is None and cell:
                        self.chars[row][cell - 1] = " "
                    elif cell + 1 < self.width and self.chars[row][cell + 1] is None:
                        self.chars[row][cell + 1] = " "
                    self.chars[row][cell] = " "
                self.chars[row][column] = char
                for cell in range(column + 1, column + size):
                    self.chars[row][cell] = None
                for cell in range(column, column + size):
                    old_bg = self.styles[row][cell][1]
                    self.styles[row][cell] = (color, background or old_bg)
                previous = column
            column += size

    def fill(self, top, left, width, height, background):
        for row in range(top, top + height):
            self.put(row, left, " " * width, background=background)

    def line(self, row, left, width, color=BORDER):
        self.put(row, left, "─" * max(0, width), color)

    def box(self, top, left, width, height, color=BORDER, background=None):
        if background:
            self.fill(top, left, width, height, background)
        self.put(top, left, "╭" + "─" * (width - 2) + "╮", color)
        self.put(top + height - 1, left, "╰" + "─" * (width - 2) + "╯", color)
        for row in range(top + 1, top + height - 1):
            self.put(row, left, "│", color)
            self.put(row, left + width - 1, "│", color)

    def number(self, row, left, value, color=GREEN):
        for index, digit in enumerate(f"{value:03d}"):
            for offset, glyph in enumerate(DIGITS[digit]):
                self.put(row + offset, left + index * 4, glyph, color)

    def bar(self, row, left, value, width, color=GREEN):
        value = min(1, max(0, value))
        count = round(value * width)
        self.put(row, left, "─" * width, GRID)
        for index in range(count):
            self.put(row, left + index, "━", _mix(BORDER, color, (index + 1) / max(1, count)))

    def render(self, color=False):
        lines = []
        for chars, styles in zip(self.chars, self.styles):
            if not color:
                lines.append("".join(char for char in chars if char is not None))
                continue
            chunks, previous = [], None
            for char, style in zip(chars, styles):
                if char is None:
                    continue
                if style != previous:
                    fg, bg = style
                    chunks.append("\033[38;2;" + ";".join(map(str, _rgb(fg)))
                                  + ";48;2;" + ";".join(map(str, _rgb(bg))) + "m")
                    previous = style
                chunks.append(char)
            lines.append("".join(chunks) + "\033[0m")
        return "\n".join(lines)


def _status(game, paused):
    reason = game.get("terminal_reason")
    if reason:
        return {"board_filled": "BOARD CLEAR", "wall": "GAME OVER · WALL", "body": "GAME OVER · BODY",
                "starvation": "RUN ENDED · STARVATION", "max_steps": "STEP LIMIT REACHED",
                "model_error": "MODEL ERROR"}.get(reason, clean_text(reason).upper()), (
                    GREEN if game.get("won") else RED if reason in {"wall", "body", "model_error"} else AMBER)
    return ("PAUSED", AMBER) if paused else ("LIVE · MODEL CONTROL", GREEN)


def _board(c, game, top, left, cell_columns=2, cell_rows=1):
    size = game["size"]
    c.box(top, left, size * cell_columns + 2, size * cell_rows + 2, BORDER, PANEL)
    for y in range(size):
        for x in range(size):
            c.put(top + 1 + y * cell_rows, left + 1 + x * cell_columns,
                  "·" + " " * (cell_columns - 1), GRID)
    body = game["body"]
    for index in reversed(range(len(body))):
        x, y = body[index]
        tone = "d4fff0" if index == 0 else _mix("236b63", GREEN, 1 - index / max(1, len(body)))
        highlight, shadow = _mix(tone, TEXT, 0.22), _mix(tone, BG, 0.25)
        for offset in range(cell_rows):
            shade = _mix(highlight, shadow, offset / (cell_rows - 1)) if cell_rows > 1 else tone
            c.put(top + 1 + y * cell_rows + offset, left + 1 + x * cell_columns,
                  "█" * cell_columns, shade)
        if index == 0:
            # A directional mark remains readable with color disabled.
            head_row = (cell_rows - 1) // 2
            head_tone = _mix(highlight, shadow, head_row / (cell_rows - 1)) if cell_rows > 1 else tone
            c.put(top + 1 + y * cell_rows + head_row, left + 1 + x * cell_columns + (cell_columns - 1) // 2,
                  ARROWS.get(game["direction"], "◆"), BG, head_tone)
    if game["food"] is not None:
        x, y = game["food"]
        food_left, food_top = left + 1 + x * cell_columns, top + 1 + y * cell_rows
        if cell_rows > 1:
            center_row = food_top + (cell_rows - 1) // 2
            center_column = food_left + (cell_columns - 1) // 2
            c.put(center_row, center_column - 1, "·◆·", AMBER)
            c.put(center_row + 1, center_column, "˙", _mix(AMBER, PANEL, 0.35))
            if cell_rows > 2:
                c.put(center_row - 1, center_column, "˙", _mix(AMBER, PANEL, 0.35))
        else:
            c.put(food_top, food_left, "◆", AMBER)


def _probabilities(c, game, top, left, width):
    decision = game.get("last_decision") or {}
    probabilities = decision.get("probabilities", {})
    available = game.get("available_directions", ())
    bar_width = max(3, width - 17)
    for index, direction in enumerate(ARROWS):
        row = top + index
        value = probabilities.get(direction)
        selected = decision.get("executed") == direction
        tone = GREEN if selected else MUTED
        if selected:
            c.fill(row, left, width, 1, "17362f")
        c.put(row, left, f"{'›' if selected else ' '} {ARROWS[direction]} {direction:<5}", tone)
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
            c.bar(row, left + 10, value, bar_width, GREEN if selected else CYAN)
            c.put(row, left + width - 6, f"{value:6.1%}", tone)
        else:
            c.put(row, left + 10, "─" * bar_width, GRID)
            c.put(row, left + width - 6,
                  "   N/A" if decision or direction not in available else "     —", MUTED)


def compose_frame(game, *, episode=1, episodes=1, paused=False, fps=8, columns=80, rows=None):
    """Expose the same canvas to live rendering and faithful screenshot exporters."""
    width = min(120, max(1, columns - 1))  # Leave the terminal's wrapping column empty.
    size = game["size"]
    wide = width >= max(76, 2 * size + 37)
    right_width = 40 if width >= 104 else 32
    left_width = width - right_width - 7
    scale = (max(1, min(3, (left_width - 4) // (size * 2),
                        (rows - 14) // size if rows is not None else 3)) if wide else 1)
    natural_height = max(30, size * scale + 14) if wide else size + 13
    height = max(1, min(natural_height, rows)) if rows is not None else natural_height
    compact = wide and height < natural_height
    c = Canvas(width, height)
    decision = game.get("last_decision") or {}
    response = decision.get("response") or {}
    usage = response.get("usage") or {}
    latency = decision.get("inference_ms")
    input_tokens, output_tokens = usage.get("input_tokens", "—"), usage.get("output_tokens", "—")
    move = decision.get("executed") or "—"
    state, state_color = _status(game, paused)
    observation = game.get("policy", {}).get("observation", "local")
    disclosure = ("静态BFS环境特征·无动作接管" if observation == "spatial"
                  else "碰撞与食物距离环境特征·无动作接管")
    pace = "UNPACED" if fps == 0 else f"≤ {fps:g} STEPS/S"
    if width < size + 4 or height < size + 4:
        c.put(0, 0, "QEV / SNAKE", GREEN)
        c.put(1, 0, f"SCORE {game['score']} · STEP {game['step']}", TEXT)
        c.put(2, 0, state, state_color)
        c.put(3, 0, "RESIZE TERMINAL", AMBER)
        c.put(4, 0, f"Need {size + 5} × {size + 4}", MUTED)
        c.put(height - 1, 0, "SPACE · N · Q quit" if width >= 18 else "Q quit", TEXT)
        return c

    c.box(0, 0, width, height)
    if wide:
        minimal = height < size + 10
        header = "Q E V  /  S N A K E"
        c.put(1 if not minimal else 0, 3, header, GREEN)
        c.put(1 if not minimal else 0, max(29, width - cell_width(state) - 3),
              state, state_color, "17312e" if state_color == GREEN else PANEL)
        if not minimal:
            c.put(2, 3, f"ARCADE LAB   /   EPISODE {episode:02d}/{episodes:02d}   /   SEED {game['seed']}", MUTED)
            c.put(2, width - cell_width(pace) - 3, pace, CYAN)
            c.line(3, 2, width - 4)
        top = 1 if minimal else 5 if compact else 6
        board_columns, board_rows = scale * 2, scale
        board_width = size * board_columns + 2
        left = 2 + (left_width - board_width) // 2
        right = width - right_width - 3
        if not minimal:
            for row in range(4, height - 4):
                c.put(row, right - 2, "│", GRID)
        if not minimal:
            c.put(4, 3, f"01 / PLAYFIELD    {size:02d} × {size:02d}", MUTED)
        _board(c, game, top, left, board_columns, board_rows)
        sidebar_top = 1 if minimal else 4 if compact else 5
        c.put(sidebar_top, right, "SCORE", MUTED)
        c.put(sidebar_top, right + 18, "LENGTH", MUTED)
        c.number(sidebar_top + 1, right, game["score"], GREEN)
        c.number(sidebar_top + 1, right + 18, game["length"], CYAN)
        c.put(sidebar_top + 5, right, f"STEP {game['step']:03d} / {game['max_steps']}   ·   {game['score']} FOOD", MUTED)
        label_row = sidebar_top + (6 if compact or minimal else 8)
        c.put(label_row, right, "LAST DECISION", MUTED)
        c.put(label_row + 1, right, f"{ARROWS.get(move, '·')}  MOVE {move}", GREEN if decision.get("executed") else MUTED)
        if decision:
            c.put(label_row + 1, right + right_width - 9,
                  f"STEP {decision.get('step_after', game['step']):03d}", MUTED)
        probability_top = label_row + (2 if compact or minimal else 3)
        _probabilities(c, game, probability_top, right, right_width)
        telemetry = probability_top + 4
        c.put(telemetry, right, f"INFERENCE  {latency:.1f} ms" if latency is not None else "INFERENCE  —", CYAN)
        c.put(telemetry + 1, right, f"INPUT {input_tokens} tokens / OUTPUT {output_tokens}", MUTED, limit=right_width)
        if not compact and not minimal:
            c.put(telemetry + 2, right, "MODEL ARGMAX / OVERRIDE OFF", MUTED)
            c.put(telemetry + 3, right, "Choice probability · not survival", MUTED)
            facts = next((item for item in decision.get("candidate_facts", [])
                          if item.get("direction") == move), None)
            if facts and telemetry + 10 < height - 4:
                c.line(telemetry + 5, right, right_width, GRID)
                c.put(telemetry + 6, right, "OBSERVATION / LAST STEP", MUTED)
                if observation == "spatial":
                    c.put(telemetry + 7, right, f"SPACE {facts['reachable_space']} cells", CYAN)
                    tail = "REACHABLE" if facts["tail_reachable"] else "UNREACHABLE"
                    c.put(telemetry + 8, right, f"TAIL  {tail}", MUTED)
                    distance = facts["food_path_distance"]
                    path = f"{distance} steps" if distance is not None else "UNREACHABLE"
                    c.put(telemetry + 9, right, f"FOOD PATH  {path}", MUTED)
                    c.put(telemetry + 10, right, f"RECENT VISITS  {facts['recent_visits']}", MUTED)
                else:
                    c.put(telemetry + 7, right, f"COLLISION  {facts['collision'] or 'NONE'}", CYAN)
                    c.put(telemetry + 8, right, f"FOOD DISTANCE  {facts['manhattan_distance']}", MUTED)
                    c.put(telemetry + 9, right, f"DISTANCE CHANGE  {facts['manhattan_change']:+d}", MUTED)
        board_bottom = top + size * board_rows + 2
        footer_top = height - 4
        if board_bottom < footer_top:
            c.put(board_bottom, 3, "BOARD", MUTED)
            c.bar(board_bottom, 10, len(game["body"]) / (size * size), max(4, left_width - 18), GREEN)
            c.put(board_bottom, left_width - 5, f"{len(game['body']) / (size * size):5.1%}", GREEN)
        if board_bottom + 1 < footer_top:
            c.put(board_bottom + 1, 3, "STEPS", MUTED)
            c.bar(board_bottom + 1, 10, game["step"] / game["max_steps"], max(4, left_width - 18), CYAN)
            c.put(board_bottom + 1, left_width - 5, f"{game['step'] / game['max_steps']:5.1%}", CYAN)
        if not minimal:
            c.line(footer_top, 2, width - 4)
            c.put(footer_top + 1, 3, "SPACE pause / resume   N step   +/− speed   Q quit", TEXT)
            c.put(footer_top + 2, 3, f"{observation.upper()} / {disclosure}", MUTED, limit=width - 6)
        else:
            c.put(height - 1, 2, " SPACE pause   N step   +/− speed   Q quit ", MUTED, limit=width - 4)
    else:
        minimal = height < size + 9
        c.put(0 if minimal else 1, 2, f"QEV / SNAKE    SCORE {game['score']:03d}", GREEN, limit=width - 4)
        if not minimal:
            c.put(2, 2, state, state_color, limit=width - 4)
        board_columns = 2 if width >= size * 2 + 4 else 1
        _board(c, game, 1 if minimal else 3, (width - size * board_columns - 2) // 2, board_columns)
        below = size + 3 if minimal else size + 5
        details = [
            (f"STEP {game['step']}/{game['max_steps']}  LENGTH {game['length']}  SEED {game['seed']}", TEXT),
            (f"MOVE {move}  ·  LAST DECISION", GREEN),
            ("  ".join(f"{ARROWS.get(key, key)} {value:.1%}" for key, value
                       in decision.get("probabilities", {}).items()), CYAN),
            (f"INFERENCE {latency:.1f} ms" if latency is not None else "INFERENCE —", MUTED),
            (f"INPUT {input_tokens} tokens / OUTPUT {output_tokens}", MUTED),
            ("SPACE pause  N step  +/− speed  Q", TEXT),
            (f"{observation.upper()} / {disclosure}", MUTED),
        ]
        for index, (text, tone) in enumerate(details[:max(0, height - below - 1)]):
            c.put(below + index, 2, text, tone, limit=width - 4)
        if minimal:
            c.put(height - 1, 2, f" STEP {game['step']} · MOVE {move} · Q quit ", MUTED, limit=width - 4)
    if decision.get("error"):
        row = height - 3 if wide and height >= size + 10 else height - 2
        c.put(row, 2, " " * (width - 4), RED)
        c.put(row, 2, "MODEL ERROR · " + clean_text(decision["error"].get("message", "Decision failed")),
              RED, limit=width - 4)
    return c


def render_frame(game, *, episode=1, episodes=1, paused=False, fps=8, color=False, columns=80, rows=None):
    return compose_frame(game, episode=episode, episodes=episodes, paused=paused,
                         fps=fps, columns=columns, rows=rows).render(color=color)
