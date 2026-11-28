"""FastAPI inference server for the BarahAI model.

The API is **stateless**: every request carries the game so far as a list of
action ids, and the server replays it with the rules engine. That keeps the
server simple, lets it run several workers, and means a restart loses nothing.

* ``POST /api/state``    board, side to move, legal moves, result
* ``POST /api/move``     the model's move (policy only, or network-guided search)
* ``POST /api/analyse``  evaluation bar value and the model's top moves
* ``POST /api/games``    record a finished game (validated by replaying it)
* ``GET  /api/model``    which model versions are loaded
* ``GET  /api/health``   liveness and readiness
* ``GET  /metrics``      Prometheus metrics
* ``GET  /``             the web page to play against the model
* ``GET  /docs``         interactive API documentation (OpenAPI)

Shadow deployment: the champion's move is always the one returned. If a
shadow model is present (the registry's challenger, or the previous champion
right after a promotion) it scores every move request too, silently; whether it
agreed with the champion and how long it took are recorded as metrics, so two
models can be compared on real traffic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from ..engine import GameState, IllegalMoveError, group_turns, square_name
from ..inference import OnnxAgent, move_probabilities
from ..search import SearchAgent

MODEL_DIR = Path(os.environ.get("BARAHAI_MODEL_DIR", "models/serve"))
DB_PATH = Path(os.environ.get("BARAHAI_DB", "data/games.db"))
STATIC = Path(__file__).parent / "static"
MAX_ACTIONS = 1500
COLOURS = ("carnelian", "turquoise")

INFERENCE = Histogram(
    "barahai_inference_seconds",
    "Time to choose a move",
    ["role", "mode"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
MOVES = Counter("barahai_moves_total", "Moves chosen by the model", ["role", "version", "mode"])
SHADOW = Counter("barahai_shadow_agreement_total", "Shadow model agreed with the champion's move", ["agree"])
MODEL_INFO = Gauge("barahai_model_info", "Loaded model versions (value is always 1)", ["role", "version"])
GAMES = Counter("barahai_games_total", "Finished games recorded", ["result", "mode"])


class Model:
    def __init__(self, role: str, version: str, path: Path) -> None:
        self.role, self.version, self.path = role, version, path
        self.net = OnnxAgent(str(path), threads=1, name=role)


def load_models(model_dir: Path) -> dict[str, Model]:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    models = {}
    for role in ("champion", "shadow"):
        entry = manifest.get(role)
        if entry:
            models[role] = Model(role, str(entry["version"]), model_dir / entry["file"])
            MODEL_INFO.labels(role, str(entry["version"])).set(1)
    if "champion" not in models:
        raise RuntimeError(f"no champion in {model_dir / 'manifest.json'}")
    return models


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, played_at REAL, model_version TEXT,"
            " human_colour TEXT, mode TEXT, winner TEXT, reason TEXT, turns INTEGER, actions TEXT)"
        )


STATE: dict = {}
DB_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["models"] = load_models(MODEL_DIR)
    init_db(DB_PATH)
    yield
    STATE.clear()


app = FastAPI(
    title="BarahAI inference API",
    description="Play Barah Goti (12 Beads) against a reinforcement-learning model.",
    version="0.3.0",
    lifespan=lifespan,
)
Instrumentator(excluded_handlers=["/metrics", "/api/health"]).instrument(app).expose(app, include_in_schema=False)


# --- schemas -----------------------------------------------------------------


class GameIn(BaseModel):
    actions: list[int] = Field(default_factory=list, max_length=MAX_ACTIONS, description="Action ids played so far")


class MoveIn(GameIn):
    mode: Literal["policy", "search"] = Field("search", description="policy: network only; search: guided search")
    simulations: int = Field(100, ge=1, le=800, description="Search simulations (search mode only)")


class GameRecordIn(GameIn):
    human_colour: Literal["carnelian", "turquoise"]
    mode: Literal["policy", "search"] = "search"


class MoveOut(BaseModel):
    action: int
    src: str
    dst: str
    captured: str | None
    notation: str
    value: float = Field(description="Position value for the side that moved, from -1 (losing) to 1 (winning)")
    model_version: str
    mode: str
    latency_ms: float
    shadow: dict | None = None


# --- helpers -----------------------------------------------------------------


def replay(actions: list[int]) -> GameState:
    s = GameState.new()
    try:
        for a in actions:
            s = s.apply(a)
    except IllegalMoveError as e:
        raise HTTPException(status_code=422, detail=f"illegal move sequence: {e}") from e
    return s


def describe_state(s: GameState, actions: list[int]) -> dict:
    states = [GameState.new()]
    for a in actions:
        states.append(states[-1].apply(a))
    turns = group_turns(states[:-1], actions) if actions else []
    legal = []
    for a in s.legal_actions():
        m = s.describe(a)
        legal.append({
            "action": a,
            "src": m.src,
            "dst": m.dst,
            "captured": m.captured,
        })
    return {
        "board": [s.piece_at(i) for i in range(25)],
        "turn": COLOURS[s.turn],
        "chain": s.chain,
        "counts": {COLOURS[0]: s.count(0), COLOURS[1]: s.count(1)},
        "quiet_turns": s.quiet,
        "turn_number": s.turn_no,
        "result": None if s.result is None else {
            "winner": None if s.result.winner is None else COLOURS[s.result.winner],
            "reason": s.result.reason,
        },
        "legal": legal,
        "history": [{"player": COLOURS[t.player], "notation": t.notation()} for t in turns],
    }


def choose_move(model: Model, s: GameState, mode: str, simulations: int) -> tuple[int, float, float]:
    """(action, value for the mover, seconds)."""
    t0 = time.perf_counter()
    if mode == "search" and len(s.legal_actions()) > 1:
        agent = SearchAgent(model.net, simulations=simulations)
        a = agent.act(s)
        value = agent.last_value
    else:
        logits, value = model.net.evaluate(s)
        probs = move_probabilities(logits, s)
        a = max(probs, key=probs.get)
    dt = time.perf_counter() - t0
    INFERENCE.labels(model.role, mode).observe(dt)
    MOVES.labels(model.role, model.version, mode).inc()
    return a, value, dt


# --- routes ------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    models = STATE.get("models", {})
    return {"status": "ok" if models else "starting", "models": {r: m.version for r, m in models.items()}}


@app.get("/api/model")
def model_info() -> dict:
    return {
        r: {"version": m.version, "file": m.path.name, "has_value_head": m.net.has_value}
        for r, m in STATE["models"].items()
    }


@app.post("/api/state")
def state(game: GameIn) -> dict:
    return describe_state(replay(game.actions), game.actions)


@app.post("/api/move", response_model=MoveOut)
def move(req: MoveIn) -> MoveOut:
    s = replay(req.actions)
    if s.is_terminal:
        raise HTTPException(status_code=409, detail="the game is over")
    models = STATE["models"]
    champ = models["champion"]
    a, value, dt = choose_move(champ, s, req.mode, req.simulations)
    shadow = None
    if "shadow" in models:
        b, b_value, b_dt = choose_move(models["shadow"], s, req.mode, req.simulations)
        SHADOW.labels(str(a == b).lower()).inc()
        shadow = {"version": models["shadow"].version, "agrees": a == b, "value": round(b_value, 4),
                  "latency_ms": round(b_dt * 1000, 2)}
    m = s.describe(a)
    sep = "x" if m.captured is not None else "-"
    return MoveOut(
        action=a,
        src=square_name(m.src),
        dst=square_name(m.dst),
        captured=None if m.captured is None else square_name(m.captured),
        notation=f"{square_name(m.src)}{sep}{square_name(m.dst)}",
        value=round(value, 4),
        model_version=champ.version,
        mode=req.mode,
        latency_ms=round(dt * 1000, 2),
        shadow=shadow,
    )


@app.post("/api/analyse")
def analyse(game: GameIn, top: int = 3) -> dict:
    s = replay(game.actions)
    if s.is_terminal:
        w = s.result.winner
        bar = 0.0 if w is None else (1.0 if w == 0 else -1.0)
        return {"carnelian_value": bar, "top_moves": []}
    champ = STATE["models"]["champion"]
    logits, value = champ.net.evaluate(s)
    probs = move_probabilities(logits, s)
    best = sorted(probs.items(), key=lambda kv: -kv[1])[: max(1, min(top, 10))]
    moves = []
    for a, p in best:
        m = s.describe(a)
        moves.append({"action": a, "from": square_name(m.src), "to": square_name(m.dst), "probability": round(p, 4)})
    return {"carnelian_value": round(value if s.turn == 0 else -value, 4), "top_moves": moves,
            "model_version": champ.version}


@app.post("/api/games", status_code=201)
def record_game(game: GameRecordIn) -> dict:
    s = replay(game.actions)
    if not s.is_terminal:
        raise HTTPException(status_code=422, detail="the game is not finished")
    human = COLOURS.index(game.human_colour)
    w = s.result.winner
    result = "draw" if w is None else ("human_win" if w == human else "model_win")
    GAMES.labels(result, game.mode).inc()
    version = STATE["models"]["champion"].version
    with DB_LOCK, sqlite3.connect(DB_PATH) as db:
        cur = db.execute(
            "INSERT INTO games (played_at, model_version, human_colour, mode, winner, reason, turns, actions)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), version, game.human_colour, game.mode,
             None if w is None else COLOURS[w], s.result.reason, s.turn_no, json.dumps(game.actions)),
        )
    return {"id": cur.lastrowid, "result": result, "model_version": version}


@app.get("/api/stats")
def stats() -> dict:
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT model_version, winner = human_colour, winner IS NULL, COUNT(*) FROM games GROUP BY 1, 2, 3"
        ).fetchall()
    out: dict = {}
    for version, human_won, draw, n in rows:
        d = out.setdefault(version, {"games": 0, "human_wins": 0, "model_wins": 0, "draws": 0})
        d["games"] += n
        key = "draws" if draw else ("human_wins" if human_won else "model_wins")
        d[key] += n
    return out


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
