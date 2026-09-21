"""Local demo API; model and game state stay authoritative on the server."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from .snake import GameCapacity, GameConflict, GameNotFound, GameStore


class CreateGame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(default=7, ge=-(2**31), lt=2**31)
    size: int = Field(default=12, ge=6, le=20)
    max_steps: int = Field(default=500, ge=1, le=2000)
    observation: Literal["local", "spatial"] = "spatial"


class StepGame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_step: int = Field(ge=0, le=2000)


def create_demo_router(agent, *, store=None):
    """Mount with app.include_router(create_demo_router(agent)).

    Reset is DELETE then POST, preserving the seed for the same initial board.
    GET ?history=true includes all retained decision traces (at most 2000).
    """
    router = APIRouter(prefix="/api/snake", tags=["snake"])
    games = store if store is not None else GameStore(agent)

    def invoke(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except GameNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except GameConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except GameCapacity as exc:
            raise HTTPException(429, str(exc)) from exc

    @router.post("/games", status_code=201)
    def create_game(request: CreateGame):
        return invoke(games.create, **request.model_dump())

    @router.get("/games/{game_id}")
    def get_game(game_id: str, history: bool = False):
        return invoke(games.get, game_id, history=history)

    @router.post("/games/{game_id}/step")
    def step_game(game_id: str, request: StepGame):
        return invoke(games.step, game_id, request.expected_step)

    @router.delete("/games/{game_id}", status_code=204)
    def delete_game(game_id: str):
        invoke(games.delete, game_id)
        return Response(status_code=204)

    return router
