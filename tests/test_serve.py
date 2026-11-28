"""Tests for the FastAPI inference server, using small untrained models."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from barahai.engine import GameState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("serve")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_bundle.py"), str(bundle)], check=True)
    app_mod = importlib.import_module("barahai.serve.app")
    app_mod.MODEL_DIR = bundle
    app_mod.DB_PATH = bundle / "games.db"
    with TestClient(app_mod.app) as c:
        yield c


def test_health_and_model(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/model").json()
    assert m["champion"]["version"] == "test-1" and m["champion"]["has_value_head"]
    assert m["shadow"]["version"] == "test-2"


def test_state_lists_legal_moves(client):
    r = client.post("/api/state", json={"actions": []}).json()
    assert r["turn"] == "carnelian" and len(r["legal"]) == 4 and r["result"] is None
    assert sum(p >= 0 for p in r["board"]) == 24


@pytest.mark.parametrize("mode", ["policy", "search"])
def test_move_is_legal_and_shadowed(client, mode):
    r = client.post("/api/move", json={"actions": [], "mode": mode, "simulations": 20})
    assert r.status_code == 200
    body = r.json()
    assert GameState.new().is_legal(body["action"])
    assert body["shadow"]["version"] == "test-2" and isinstance(body["shadow"]["agrees"], bool)


def test_illegal_history_is_rejected(client):
    assert client.post("/api/move", json={"actions": [999]}).status_code == 422
    assert client.post("/api/move", json={"actions": [], "simulations": 5000}).status_code == 422


def test_full_game_is_recorded_and_counted(client):
    actions, s = [], GameState.new()
    while not s.is_terminal:
        a = client.post("/api/move", json={"actions": actions, "mode": "policy"}).json()["action"]
        actions.append(a)
        s = s.apply(a)
    assert client.post("/api/move", json={"actions": actions}).status_code == 409
    r = client.post("/api/games", json={"actions": actions, "human_colour": "turquoise", "mode": "policy"})
    assert r.status_code == 201
    assert client.get("/api/stats").json()["test-1"]["games"] == 1
    metrics = client.get("/metrics").text
    assert "barahai_games_total" in metrics and "barahai_shadow_agreement_total" in metrics


def test_analyse_and_web_page(client):
    r = client.post("/api/analyse", json={"actions": []}).json()
    assert -1 <= r["carnelian_value"] <= 1 and len(r["top_moves"]) == 3
    assert "Barah Goti" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
