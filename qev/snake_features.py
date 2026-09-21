"""Observable Snake features and a declared, deterministic imitation teacher.

The teacher sees only fields included in the model request. Static connectivity
is a useful space estimate, not a proof that future moving-body play is safe.
No function in this module executes or substitutes the model's chosen action.
"""

from __future__ import annotations

from collections import Counter, deque

FEATURE_VERSION = "snake-observable-bfs-v1"
RECENT_WINDOW = 32
_VECTORS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ORDER = {direction: index for index, direction in enumerate(("UP", "DOWN", "LEFT", "RIGHT"))}


def _distances(start, blocked, size):
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _VECTORS:
            cell = x + dx, y + dy
            if (0 <= cell[0] < size and 0 <= cell[1] < size
                    and cell not in blocked and cell not in distances):
                distances[cell] = distances[(x, y)] + 1
                queue.append(cell)
    return distances


def candidate_features(body, food, size, candidates, recent_heads=()):
    """Return new candidate dicts, simulating exactly one legal Snake step.

    ``reachable_space`` counts the new head plus empty cells connected to it,
    with the new body (including tail) blocked. ``tail_reachable`` separately
    permits the new tail as an endpoint. ``food_path_distance`` uses that fixed
    new body; eating has distance zero and an inaccessible food has None.
    ``recent_visits`` counts visits to the target among the last 32 heads.
    Input lists/dicts are not modified. No future food RNG is inspected.
    """
    body = [tuple(cell) for cell in body]
    food = tuple(food) if food is not None else None
    visits = Counter(tuple(cell) for cell in list(recent_heads)[-RECENT_WINDOW:])
    result = []
    for original in candidates:
        move = dict(original)
        target = tuple(move["next_cell"])
        move.update(reachable_space=0, tail_reachable=False, food_path_distance=None,
                    recent_visits=visits[target])
        if move["collision"] in (None, "none"):
            after = [target, *body]
            if not move["eats_food"]:
                after.pop()
            distances = _distances(target, set(after[1:]), size)
            tail = after[-1]
            # A blocked tail is reachable iff some reachable cell borders it;
            # no second search or passage through the tail is needed.
            tail_reachable = tail in distances or any(
                (tail[0] + dx, tail[1] + dy) in distances for dx, dy in _VECTORS
            )
            move.update(reachable_space=len(distances), tail_reachable=tail_reachable,
                        food_path_distance=0 if move["eats_food"] else distances.get(food))
        result.append(move)
    return result


def teacher_choice(candidates, *, length, size, heading):
    """Choose from visible feature values only; None means all moves collide.

    Prefer a route back to the moving tail and enough static free space, then
    follow a short food path while penalizing repeatedly visited cells. If food
    is cut off, prefer unvisited connected space. Ties favor continuing straight
    and then a fixed absolute direction, never the incoming candidate order.
    This is an explicit heuristic policy, not an optimal-action oracle.
    """
    legal = [move for move in candidates if move["collision"] in (None, "none")]
    if not legal:
        return None

    def rank(move):
        won = move["eats_food"] and length + 1 == size * size
        space_needed = min(length + int(move["eats_food"]), size * size - length)
        enough_space = move["reachable_space"] >= space_needed
        distance = move["food_path_distance"]
        food_access = distance is not None
        progress = -(distance + 2 * move["recent_visits"]) if food_access else -move["recent_visits"]
        return (won, move["tail_reachable"], enough_space, food_access, progress,
                move["reachable_space"], move["eats_food"], move["direction"] == heading,
                -_ORDER[move["direction"]])

    return max(legal, key=rank)["direction"]
