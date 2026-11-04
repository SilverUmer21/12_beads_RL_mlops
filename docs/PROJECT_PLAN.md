# BarahAI project plan

This file records the research, measurements, decisions and plan for turning the
Barah Goti game into an RL agent with a full MLOps pipeline. It is the working
reference for the project: what we found, what we recommend, and what we are
actually building.

Last updated: November 2026.

**Contents**

1. [Where the project stands](#1-where-the-project-stands)
2. [Course requirements (Phase 1)](#2-course-requirements-phase-1)
3. [Research findings](#3-research-findings)
4. [Decisions](#4-decisions)
5. [Architecture](#5-architecture)
6. [The plan: 4 stages in 4 weeks](#6-the-plan-4-stages-in-4-weeks)
7. [How we work](#7-how-we-work)
8. [Additions we considered](#8-additions-we-considered)
9. [Verified tool versions and gotchas](#9-verified-tool-versions-and-gotchas)
10. [Open questions](#10-open-questions)
11. [Sources](#11-sources)

---

## 1. Where the project stands

**Already built** (see the main [README](../README.md)):

- A pure-Python rules engine with bitboards, a fixed space of 201 atomic actions,
  `action_mask()`, immutable states and the 8 board symmetries.
- Three opponents: Random, Greedy (rule-based) and Alpha-beta search.
- A pygame desktop client, and 39 tests, including differential tests against a naive engine.
- `scripts/arena.py` for bot-vs-bot tournaments.

**To build:** the RL agent and everything around it: data, training, tracking,
registry, serving, monitoring and deployment.

**Team and machine:**

| | |
|---|---|
| Team | Solo |
| Time | About 4 weeks |
| Laptop | Ryzen 7 (8 cores), 16 GB DDR5, RTX 3050 Ti (4 GB) |
| OS | Windows with WSL2 (Debian) |
| Cloud | Own AWS account (new-account credits) |

The original proposal is *BarahAI: Reinforcement Learning Game Agent with an Automated
MLOps Pipeline*. It covers Maskable PPO, training against random and rule-based bots and
then self-play, MLflow, DVC, GitHub Actions, Flask, Docker and AWS EC2.

---

## 2. Course requirements (Phase 1)

The TA's deliverables are a template written for the whole class. We pick **one** of the
three pipeline types.

| Requirement | What it means | How we meet it |
|---|---|---|
| Dataset & versioning | Raw and processed datasets tracked with DVC | We **generate** datasets: expert games (raw), encoded positions split by game (processed), and a puzzle suite. All tracked with DVC on S3 |
| Pipeline type 1: Standard models | Preprocessing, training and validation as repeatable pipeline runs | ✅ Our pipeline: `generate_data → preprocess → pretrain → train_rl → evaluate → register` in `dvc.yaml` |
| Pipeline type 2: RAG | Ingestion, chunking, embeddings, vector DB | Not applicable |
| Pipeline type 3: Agentic | Multi-agent loops | Not applicable |
| Experiment tracking | Hyperparameters, metrics and artifacts | MLflow: every stage logs to one tracking server |
| Model registry | Model artifacts with version tags | MLflow Model Registry: versions, tags (data version, git commit, scores), `@champion` alias |

**Submission:** a public GitHub repo (no `.zip`), plus a one-page PDF with the repo link
and a short summary. The summary must be written by us, not a multi-page AI write-up.

**Why we generate data:** pure RL has no dataset, which would look weak against this
rubric. Learning from expert games first and then improving with RL fills that gap. It is
the same approach AlphaGo used (copy expert moves, then improve by self-play), and it
should also make PPO learn faster.

---

## 3. Research findings

All measurements below were taken on a 4-core cloud machine. The laptop should be
similar or faster per core.

### 3.1 Speed measurements

| Measurement | Result |
|---|---|
| Time per move: Greedy | 0.2 ms |
| Time per move: Alpha-beta depth 2 | 2 ms |
| Time per move: Alpha-beta depth 4 | 27 ms (slowest 5%: 90 ms) |
| Game length | 37 turns (random vs random) to 64 turns (greedy vs alpha-beta) |
| MaskablePPO, 16 games batched in one process | ~1,750 agent moves/s on **one core** |
| MaskablePPO, 1 game | ~1,000 moves/s |
| MaskablePPO, games spread over 3 processes | ~890 moves/s (slower: the action mask crosses processes every move) |
| Where training time goes | ~75% playing games, ~25% network updates |

At this speed, 10 million training moves take about 1.5–2 hours on one core
(roughly 300k–500k games).

### 3.2 Laptop vs Google Colab

**Recommendation: train on the laptop's CPU.**

- Stable-Baselines3 warns that PPO with a standard (non-image) network is "primarily
  intended to run on the CPU". Our numbers agree: Python running the game is the
  bottleneck, not neural-network math.
- One run uses about one core, so the 8-core Ryzen can run **4–6 experiments at once**.
- Colab's free tier gives about 2 virtual CPUs, a 12-hour session limit, an idle
  disconnect after about 90 minutes and no guaranteed GPU. It also can't reach an MLflow
  server on the laptop.
- The **RTX 3050 Ti** is not needed for PPO. It becomes useful for the bonus work
  (transformer training, MCTS with batched evaluation). 4 GB is plenty at this board size.

### 3.3 RL approach review

**Prototype result:** a throwaway MaskablePPO setup, run for under 5 minutes on CPU.
Evaluation sampled moves from the policy.

| Training | vs Random | vs Greedy | vs Alpha-beta d2 |
|---|---|---|---|
| Untrained | 51 W, 1 D, 48 L | 0 W, 0 D, 50 L | — |
| 150k moves vs Random (49 s) | 100 W, 0 D, 0 L | 0 W, 23 D, 27 L | — |
| + 300k moves vs Greedy (3 min) | 98 W, 2 D, 0 L | 7 W, 40 D, 3 L | 0 W, 20 D, 0 L |

**Verdict: keep Maskable PPO.** It learns quickly and the engine is already set up for
it. What needs to change is around the algorithm:

1. **Board from the mover's side.** Rotate the board 180° for player 1 (the engine's
   `SYM_SQ` / `SYM_DIR` tables) and pick the agent's colour at random each game.
   One network then learns both colours.
2. **Let the agent see the draw rules.** Threefold repetition and the 40-quiet-turn
   rule depend on history, so the observation includes the quiet-turn count and a
   repetition count.
3. **The draw problem.** The agent learns "don't lose" much faster than "win" (40 of
   50 games drawn against Greedy). Experiments to run: a small draw penalty, and a bonus
   for being ahead on beads (potential-based shaping) that fades to zero. Track the draw
   rate as a main metric.
4. **Opponent mix, not just the latest self.** Go Random → Greedy → Alpha-beta d1/d2,
   then a pool of saved past versions plus the fixed bots, chosen each game. Alpha-beta
   d4 is too slow to train against, so it is for evaluation only.
5. **Meaningful evaluation.** Fixed-move play replays the same game every time, and the
   opening is very narrow (4, 5, 6, 12, 19 possible lines at depths 1–5). Use random
   openings with fixed seeds, play both colours, and count wins, draws and losses.
6. **Statistical promotion** (later): a sequential probability ratio test (SPRT), as
   Stockfish's Fishtest uses, instead of a fixed win-rate threshold.
7. **Expect a strength ceiling.** A network without lookahead will likely level off
   around alpha-beta depth 2–3. Beating depth 4 probably needs search at play time
   (MCTS using the network). This is a stretch goal.
8. **Reuse the `Agent` interface.** A `NeuralAgent.act(state)` plugs into `arena.py`,
   the pygame client and the web app.

### 3.4 MLOps stack review

| Piece | Verdict | Notes |
|---|---|---|
| Gymnasium + sb3-contrib | Keep | `check_env` in CI; batch games in one process (`DummyVecEnv`) |
| MLflow | Keep | Log win/draw/loss per opponent, draw rate, puzzle score, sample games as JSON. Registry aliases `@champion` / `@challenger` (old "stages" are deprecated) |
| DVC | Keep | Pipeline (`dvc.yaml` + `params.yaml`), generated datasets, puzzle suite. MLflow is the source of truth for models. lakeFS took over DVC in Nov 2025; still open source, no breaking changes |
| GitHub Actions | Keep | CI: lint, tests, `check_env`, short smoke training, puzzle gate, benchmarks. CD: build image and deploy |
| Flask + Docker | Keep | Serve with **ONNX Runtime**, not PyTorch: tested image 399 MB, about 40 MB RAM |
| AWS | Keep | S3 for DVC data, EC2 for the app. New accounts (after July 15, 2025) get $100 + up to $100 in credits on a 6-month free plan. Set a budget alarm first |
| Monitoring | Make concrete | Log every game with model version and response time; `/metrics` for Prometheus; shadow deployment data |

---

## 4. Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Maskable PPO stays the main algorithm | Prototype learns quickly; matches the proposal; supported by sb3-contrib |
| 2 | Generate expert-game datasets and pretrain before PPO | Meets the dataset requirement and speeds up learning |
| 3 | Split processed data **by game**, not by position | Otherwise positions from one game land in both train and validation, and the score is fake |
| 4 | Train on the laptop CPU, not Colab | CPU-bound workload; 8 cores beat Colab's ~2 |
| 5 | Serve with ONNX Runtime | Small image and memory footprint on a small EC2 instance |
| 6 | AI writes the HTML/JS page | Front end isn't MLOps work; we write the API |
| 7 | AI writes most code for now; every PR is reviewed and understood before merge | Time is short; understanding the foundation is the priority |
| 8 | Security additions skipped for now | Time; revisit after the core is done |
| 9 | Production-grade serving only if time is left | Least important for the grade |

**Additions chosen:**

- Puzzle suite as a CI gate
- Benchmarks in CI
- Publish the engine and environment to PyPI
- Docs site
- Evaluation bar + replay viewer
- Human leaderboard
- AI running in the browser
- Shadow deployment

---

## 5. Architecture

### 5.1 Phase 1 pipeline (`dvc.yaml`)

```
generate_data ──► preprocess ──► pretrain ──► train_rl ──► evaluate ──► register
 expert games     encode +        copy expert   MaskablePPO   vs bots +     MLflow registry
 (raw JSONL)      split by game   moves         from pretrained puzzle score  version + tags
                  (train/val)     (val accuracy) weights                      + @champion
```

- `dvc repro` runs the whole pipeline; changing one setting in `params.yaml` re-runs
  only the affected stages.
- Raw data, processed data and puzzles are DVC outputs stored in S3.
- Every stage logs to the same MLflow server.

### 5.2 Full system (later stages)

```
LAPTOP (WSL2)                                   GITHUB                        AWS
dvc repro → MLflow (registry @champion)         push → Actions CI             S3: DVC data + models
         → export ONNX → dvc push ───────────►  tests, puzzle gate,           EC2: Docker container
                                                benchmarks                       Flask + ONNX Runtime
                                                deploy: dvc pull → build ──►     champion + shadow challenger
                                                → push image → SSH to EC2        game logs, leaderboard,
                                                                                 /metrics
Players ◄──────────── web page (evaluation bar, replay viewer, optional in-browser AI) ◄──┘
```

### 5.3 Planned folder layout

```
src/barahai/
  engine/        existing rules engine
  agents/        existing bots + neural.py (NeuralAgent, ONNX)
  data/          generate.py, preprocess.py, puzzles.py
  rl/            features.py, env.py, opponents.py, pretrain.py, train.py,
                 evaluate.py, register.py, export.py
  web/           app.py, templates/, static/ (AI-written page)
params.yaml
dvc.yaml
Dockerfile
.github/workflows/   ci.yml, deploy.yml, release.yml, docs.yml
docs/adr/            architecture decision records
docs/devlog.md       weekly notes
```

---

## 6. The plan: 4 stages in 4 weeks

### Stage 0: setup (days 1–2)

- GitHub Project board, one milestone per stage, all issues created up front.
- Package extras in `pyproject.toml` (`rl`, `serve`), `.gitignore` (no keys, no `mlflow.db`).
- CI skeleton: ruff and pytest on every push.
- DVC with an S3 remote; MLflow server with a SQLite database.
- Check for secrets, then make the repo public.

**Done when:** CI is green on a push, `dvc push` works, and the MLflow UI opens.

### Stage 1: data and environment (week 1)

- `generate_data`: about 5,000 alpha-beta games with random openings → `data/raw/games.jsonl`.
- `preprocess`: encode each position, split by game → `data/processed/train.npz`, `val.npz`.
- `BeadsEnv`: Gymnasium environment with mover's-side view, draw-rule features,
  opponent inside the environment, and tests (`check_env`, mask matches legal moves).
- Puzzle suite: positions where deep alpha-beta finds one clearly best move → `data/puzzles.jsonl`.
- `dvc.yaml` stages wired together.

**Done when:** `dvc repro` builds all three datasets and `pytest` passes.

### Stage 2: training, tracking, registry (week 2) → Phase 1 submission

- `pretrain`: supervised copying of expert moves, with validation accuracy.
- `train_rl`: PPO from pretrained weights; easy → hard opponents, then self-play pool.
  MLflow comparison: from scratch vs pretrained.
- `evaluate`: win/draw/loss against each bot, plus puzzle score.
- `register`: MLflow version, tags (data version, git commit, scores), promotion rule,
  `@champion` alias.
- Puzzle suite as a CI gate: CI pulls the champion with `dvc pull` and fails if the
  puzzle score drops.

**Done when:** everything in section 2 works end to end. We write the one-page PDF.

### Stage 3: serving and product features (week 3)

- ONNX export → Flask API → Docker → EC2.
- Web page (AI-written) with an evaluation bar and a replay viewer.
- AI in the browser (onnxruntime-web) as a toggle.
- Human leaderboard: Elo for players and model versions (SQLite).
- Shadow deployment: the challenger runs silently on every move. Log agreement with the
  champion, the difference in evaluations, and both response times.

**Design catch:** in browser mode the server never sees the move, so there is no shadow
data for those games. Server mode is the default; browser mode is optional.

**Done when:** a public link where people can play, and a shadow-deployment report.

### Stage 4: engineering polish (week 4)

- Benchmarks in CI: fail if the engine gets more than 10% slower.
- Publish to PyPI from GitHub Actions on a version tag.
- Docs site with MkDocs Material on GitHub Pages: rules, architecture, ADRs, results.
- README: results tables, MLflow screenshots, demo GIF, live link.
- If time is left: production serving (HTTPS, zero-downtime deploys).

### If we fall behind, cut in this order

1. Production serving
2. Human leaderboard
3. AI in the browser
4. PyPI

Stages 1 and 2 are never cut: they are the grade.

---

## 7. How we work

**Tracking.** GitHub Project board with columns Backlog → Research → In progress →
Review → Done. Every task is an issue with a goal, research notes and "done when".
Labels: `stage-1` … `stage-4`, `research`, `feature`, `bug`, `docs`.

**For every issue:**

1. Research spike (30–60 min): read the docs, write the chosen approach in the issue.
2. Big decisions go in `docs/adr/` as short decision records.
3. Branch → AI writes the code → review the PR line by line. **Don't merge anything you
   can't explain out loud.**
4. CI passes → merge → card to Done.

**Weekly:** a 10-minute retro in `docs/devlog.md`: done, blocked, next.

**Commits** are made under SilverUmer21 only.

**When stuck:** read the last line of the error, print the variables, check the docs.
After 20 minutes, ask AI for a hint before asking for code.

### Learning path (reference)

The original 24-session learning map, kept for self-study alongside the stages:

| Week | Topics |
|---|---|
| 1 | Gymnasium environments, encoding the board, the real `BeadsEnv` |
| 2 | PPO in plain words, evaluation, `params.yaml` + MLflow, learning curves via callbacks |
| 3 | DVC pipelines and experiments, opponent curriculum, AWS + S3 |
| 4 | Self-play, ONNX export, champion vs challenger + registry |
| 5 | Flask, game API, Docker, logging + `/metrics` |
| 6 | GitHub Actions CI, EC2, auto-deploy, Grafana |

Each session follows: concept → syntax on a small example → write the real piece →
review → add to a personal cheat sheet. Each session starts with a 5-minute warm-up:
retype the last example from memory.

---

## 8. Additions we considered

Status: ✅ chosen · 🕒 maybe later · ❌ skipped

### Testing the ML

| Idea | What it is | Status |
|---|---|---|
| Puzzle suite as a CI gate | Positions with a known best move; CI fails if the score drops | ✅ |
| Property-based tests | Hypothesis generates random positions and checks rules that always hold | 🕒 |
| Mutation testing | mutmut breaks code on purpose to check the tests notice | 🕒 |
| Reproducibility test | Retrain from a commit + data version, check results match | 🕒 |

### Engineering quality

| Idea | What it is | Status |
|---|---|---|
| Benchmarks in CI | Fail if the engine slows down by more than 10% | ✅ |
| Publish to PyPI | `pip install barahai`, registered as a Gymnasium/PettingZoo environment | ✅ |
| Docs site | MkDocs Material on GitHub Pages | ✅ |
| Lint, types, locked deps | ruff, pyright, pre-commit, `uv` | 🕒 |
| Dev container | One config for an identical dev setup | 🕒 |

### Serving and performance

| Idea | What it is | Status |
|---|---|---|
| KV-cache transformer | A small GPT that reads the game as a sequence of moves; per-game KV cache and a shared prefix cache for common openings; benchmark response time vs move number | 🕒 bonus candidate |
| INT8 quantization | Smaller, faster ONNX model; compare win rate | 🕒 |
| Request batching | Several players' boards in one forward pass; load test | 🕒 |
| Distillation | Small fast model copies a big strong one | 🕒 |
| Rust/Numba engine core | Faster move generation, checked by existing tests | 🕒 |
| GPU batched environment | Thousands of games at once as tensors | 🕒 |
| Symmetry augmentation | Use the 8 board symmetries to learn from fewer games | 🕒 |

### Algorithms

| Idea | What it is | Status |
|---|---|---|
| Endgame tablebase | Solve all positions with few beads exactly; measure "perfect move" accuracy | 🕒 bonus candidate |
| MCTS with tree reuse | Search at play time; likely beats alpha-beta d4; uses the GPU | 🕒 |
| League training | Pick self-play opponents by Elo | 🕒 |
| Exploiter agent | Train an agent only to beat the champion, to find weaknesses | 🕒 |
| Difficulty levels | Deliberately weaker or stronger play for the website | 🕒 |

### MLOps

| Idea | What it is | Status |
|---|---|---|
| Shadow deployment | Challenger runs silently next to the champion on real games | ✅ |
| Canary + automatic rollback | 10% of games to the new model; switch back if metrics worsen | 🕒 |
| SPRT gating + Elo with confidence intervals | Statistical champion promotion | 🕒 |
| Drift detection → retraining | Detect new human play styles and trigger retraining | 🕒 |
| Optuna + MLflow | Automatic hyperparameter search with early stopping | 🕒 |
| Cost and energy tracking | CodeCarbon; AWS cost per 1,000 games | 🕒 |
| Model card + reproducibility record | Seeds, image digest, data version, commit per model | 🕒 |
| Infrastructure as code | Terraform for EC2 and S3 | 🕒 |

### Security

| Idea | What it is | Status |
|---|---|---|
| OIDC from GitHub to AWS | No long-lived keys in GitHub secrets | ❌ for now |
| Image scanning + signing | Trivy, cosign | ❌ for now |
| Hardened container | Non-root user, health check, multi-stage build | ❌ for now |

### Product features

| Idea | What it is | Status |
|---|---|---|
| Evaluation bar + replay viewer | Step through games with the AI's evaluation | ✅ |
| Human leaderboard | Elo for players and each model version | ✅ |
| AI in the browser | onnxruntime-web, zero server cost | ✅ |
| "AI's top 3 moves" | Show the moves the model considered | 🕒 |
| Production serving | HTTPS with Caddy, zero-downtime deploys, pydantic + OpenAPI, tracing + SLOs | 🕒 end, if time |
| LLM coach | Explains moves using engine facts; MLflow tracing | 🕒 (API cost) |

### Not a fit

Speculative decoding and FlashAttention (for long text generation, not a 25-square board),
Kubernetes (too heavy for a small server), feature store (no tabular features).

---

## 9. Verified tool versions and gotchas

These versions were installed and the example code for each was run successfully:

| Tool | Version tested |
|---|---|
| sb3-contrib / Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.4.0 |
| PyTorch (CPU) | 2.14.1 |
| MLflow | 3.17.2 |
| DVC | 3.67.1 |
| Flask | 3.1.3 |
| ONNX / ONNX Runtime | 1.23.2 / 1.29.0 |
| Docker | 29.9.1 |

**Gotchas found while testing:**

- **YAML:** `3e-4` is read as a string. Write `0.0003`.
- **`model.predict`** returns a NumPy array; wrap it with `int(action)`.
- **ONNX export:** PyTorch 2.9+ uses a new exporter. Pass `external_data=False` to get
  one self-contained `.onnx` file instead of `.onnx` + `.onnx.data`. The old exporter
  (`dynamo=False`) still works but prints a removal warning.
- **MLflow registry** needs a database backend: run
  `mlflow server --backend-store-uri sqlite:///mlflow.db`.
- **`mlflow.onnx.log_model(..., name="model", registered_model_name=...)`** logs and
  registers in one call; load with `models:/<name>@champion`.
- **DVC queued experiments** run in temporary folders, so a relative SQLite path would
  create a separate MLflow database per experiment. Use the server URL
  (`http://127.0.0.1:5000`).
- **AWS profile for DVC:** use `dvc remote modify --local storage profile <name>` so
  the profile stays out of the committed config and CI can use secrets instead.
- **Gunicorn:** use 1 worker with several threads, because live games and Prometheus
  counters live in memory.
- **GitHub Actions:** Node 20 actions were removed in 2026; use `actions/checkout@v5`
  and `actions/setup-python@v6`. Install CPU-only PyTorch in CI with
  `--index-url https://download.pytorch.org/whl/cpu`.
- **Docker Hub** can rate-limit pulls; `mirror.gcr.io/library/python` is a fallback mirror.
- **WSL2:** the Windows browser can open WSL's `localhost:5000`. Wrap training code in
  `if __name__ == "__main__":`.
- **GHCR** image names must be lowercase.

---

## 10. Open questions

1. When is Phase 1 due? If sooner than 2 weeks, Stages 1–2 get compressed.
2. What do Phases 2 and 3 cover (probably deployment and monitoring)?
3. TA: does an RL project with generated expert data fit the "Standard Models" pipeline?
4. Sir: which additions count for the bonus (KV-cache transformer, tablebase, shadow
   deployment, puzzle gate)?

---

## 11. Sources

- [SB3 on-policy source (CPU warning)](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html)
- [sb3-contrib changelog](https://sb3-contrib.readthedocs.io/en/master/misc/changelog.html)
- [SB3: exporting models (ONNX)](https://stable-baselines3.readthedocs.io/en/v2.8.0/guide/export.html)
- [PettingZoo: SB3 Maskable PPO for Connect Four](https://pettingzoo.farama.org/tutorials/sb3/connect_four/)
- [Colab free tier guide](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)
- [Colab hardware spec](https://saturncloud.io/blog/whats-the-hardware-spec-for-google-colaboratory/)
- [MLflow model registry](https://mlflow.org/docs/latest/ml/model-registry)
- [DVC joins lakeFS FAQ](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/)
- [AWS Free Tier credits announcement](https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/)
- [GitHub: self-hosted runner security](https://docs.github.com/en/enterprise-server@2.22/actions/automating-your-workflow-with-github-actions/about-self-hosted-runners)
- [MCTS only at test time (arXiv 2204.13307)](https://arxiv.org/pdf/2204.13307)
- [Warm-start AlphaZero (arXiv 2004.12357)](https://arxiv.org/pdf/2004.12357)
- [SPRT (Chess Programming Wiki)](https://chessprogramming.org/SPRT)
- [Fishtest FAQ](https://github.com/official-stockfish/fishtest/wiki/Fishtest-faq)
