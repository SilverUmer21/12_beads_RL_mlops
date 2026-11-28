# Contributing to BarahAI

This guide gets you from zero to your first merged pull request, then lists the open
tasks you can pick up. Read the [README](../README.md) first for what the project is.

## 1. One-time setup

You need Python 3.12, Git and the GitHub CLI (`gh`). Everything below runs in a terminal
(PowerShell on Windows).

1. Accept the collaborator invite (GitHub email or https://github.com/notifications).
2. Clone the repository and go into it:

   ```bash
   git clone https://github.com/SilverUmer21/12_beads_RL_mlops.git
   ```
   ```bash
   cd 12_beads_RL_mlops
   ```

3. Create a virtual environment and install the project with the tools you need:

   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\python -m pip install -e .[game,dev,serve]
   ```

   Add `rl` to the list (`.[game,dev,serve,rl]`) only if you will train models; it pulls
   in PyTorch, MLflow and DVC (about 2 GB).

4. Check that everything works:

   ```bash
   .venv\Scripts\python -m pytest
   ```

   Every test should pass. Tests that need the `rl` extra are skipped without it.

5. Set your Git name and email (once per computer):

   ```bash
   git config --global user.name "Your Name"
   ```
   ```bash
   git config --global user.email "you@example.com"
   ```

## 2. How we work

Nobody pushes to `main` directly. Every change goes through a pull request (PR).

1. Pick a task from section 4 and comment on its GitHub issue so nobody duplicates it.
2. Update `main` and create a branch named after the task:

   ```bash
   git switch main
   ```
   ```bash
   git pull
   ```
   ```bash
   git switch -c load-test
   ```

3. Make the change. Keep commits small, with a message saying what changed and why.
4. Before pushing, run the same checks CI runs:

   ```bash
   .venv\Scripts\ruff check .
   ```
   ```bash
   .venv\Scripts\python -m pytest
   ```

5. Push the branch and open a PR:

   ```bash
   git push -u origin load-test
   ```
   ```bash
   gh pr create --fill
   ```

6. In the PR description write what you did, how you tested it, and `Closes #<issue>`.
7. Wait for CI to go green and for a review, fix anything raised, then it gets merged.

Rules: no secrets in commits (AWS keys, passwords, `.env` files), no large files (data and
models go through DVC, not Git), and don't merge anything you can't explain.

## 3. Where things are

| Path | What it is |
|---|---|
| `src/barahai/engine/` | Game rules (pure Python) |
| `src/barahai/agents/` | Bots: random, greedy, alpha-beta; `make_agent("onnx:...")` loads a model |
| `src/barahai/serve/` | FastAPI server (`app.py`) and the web page (`static/`) |
| `src/barahai/search.py` | Network-guided tree search used when the model plays |
| `src/barahai/rl/`, `src/barahai/data/` | Training pipeline stages |
| `compose.yaml`, `Dockerfile`, `deploy/` | Containers, proxy, Prometheus and Grafana configs |
| `.github/workflows/` | CI, Deploy, Diagram |
| `tests/` | All tests |

Run the server locally without Docker, using two small test models:

```bash
.venv\Scripts\python scripts\make_test_bundle.py models\serve
```
```bash
.venv\Scripts\python -m uvicorn barahai.serve.app:app --port 8000
```

Then open http://127.0.0.1:8000 (game) and http://127.0.0.1:8000/docs (API).
`make_test_bundle.py` needs the `rl` extra; if you don't have it, ask for a copy of
`models/serve`, or run `dvc pull models/serve` if you have AWS access.

## 4. Tasks

Each task below has a GitHub issue. Pick one, comment on the issue, follow section 2.

### A. Load test the live service (k6) (issue #23)

**Goal:** measure how the deployed API behaves under load and record the numbers.

1. Install k6 (https://grafana.com/docs/k6/latest/set-up/install-k6/).
2. Write `scripts/loadtest.js`: each virtual user starts a game with
   `POST /api/state {"actions": []}`, then loops: `POST /api/move` (mode `search`),
   appends the returned `action` to its list, until `/api/state` reports a result.
3. Add thresholds: error rate below 1%, p95 latency of `/api/move` below 500 ms.
4. Run it against the live site with 1, 5 and 10 virtual users for 2 minutes each.
5. Write `docs/load-test.md`: requests per second, p50/p95/p99 latency, error rate, and a
   screenshot of the Grafana dashboard during the test.

**Done when:** the script and the results document are merged.

### B. Shadow deployment report (issue #24)

**Goal:** show, with real games, how the champion (v2) and the shadow model (v1) compare.

1. Play at least 20 games on the live site (both colours, both model modes). Ask friends.
2. From Grafana (`/grafana`) and `GET /api/stats`, collect: number of moves, shadow
   agreement rate, latency of each model, game results.
3. Write `docs/shadow-report.md`: what shadow deployment is, the numbers, a Grafana
   screenshot, and a conclusion (keep v2 as champion or not, and why).

**Done when:** the report is merged.

### C. Engine speed benchmark in CI (issue #25)

**Goal:** CI fails if a change makes the rules engine more than 10% slower.

1. Write `scripts/benchmark.py`: time random games for a few seconds and print actions per
   second (see `engine_speed()` in `scripts/arena.py`).
2. Save a baseline number in `benchmarks/baseline.json`.
3. Add a CI step that runs the benchmark and fails if the result is below 90% of the
   baseline. CI machines vary, so compare against a baseline measured on CI itself (print
   the number from one CI run and use that).

**Done when:** the benchmark runs in CI and a deliberately slowed engine makes it fail
(test that on your branch, then revert).

### D. Replay viewer (issue #26)

**Goal:** step through a recorded game on the web page with the evaluation bar.

1. Add `GET /api/games/{id}` in `src/barahai/serve/app.py` returning the stored actions
   (games are in the SQLite table `games`).
2. In `src/barahai/serve/static/`, add a "Replay" mode: load a game id, then
   Previous/Next buttons that move through the actions, calling `/api/state` and
   `/api/analyse` for each step.
3. Add a test in `tests/test_serve.py` for the new endpoint.

**Done when:** a finished game can be replayed move by move on the page, and tests pass.

### E. Human leaderboard (issue #27)

**Goal:** a simple leaderboard of players against each model version.

1. Add an optional `player` name to `POST /api/games` (validate: 1-20 letters/digits).
2. Add `GET /api/leaderboard`: per player, games, wins, draws, losses, score.
3. Show it on the web page below the move list, and add tests.

**Done when:** names appear on the page after finished games, and tests pass.
