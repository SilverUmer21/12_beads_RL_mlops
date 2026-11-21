# BarahAI · Barah Goti (12 Beads)

Barah Goti (بارہ گوٹی, "twelve beads"), also called Bara Tehni, 12 Guti or Baro Guti, is the
village board game of Punjab, Sindh, Bihar and Bengal, rebuilt in Python. It has a
rules engine, three computer opponents and a pygame desktop client drawn entirely
in code: no image or audio files.

![Barah Goti mid-game: a chain capture in progress](docs/screenshot.png)

It also has a reinforcement learning agent with a full MLOps pipeline: versioned datasets
(DVC), a repeatable training pipeline, experiment tracking and a model registry (MLflow).
See [Phase 1: the MLOps pipeline](#phase-1-the-mlops-pipeline) below and the
[project plan](docs/PROJECT_PLAN.md) for the research, decisions and roadmap.

This folder replaces the old C++/SFML version in [`SilverUmer21/twelve_beads`](https://github.com/SilverUmer21/twelve_beads)
(see [what was wrong with it](#autopsy-of-the-legacy-c-version)).

```
barahai/
├── src/barahai/
│   ├── engine/     rules engine: pure Python, zero dependencies
│   ├── agents/     Random, Greedy (rule-based), Alpha-beta search
│   ├── game/       pygame-ce client: art, sound, app, search worker
│   ├── data/       pipeline stages: generate, preprocess, puzzles
│   └── rl/         features, Gymnasium env, policy, pretrain, train, evaluate, register
├── params.yaml     every pipeline setting
├── dvc.yaml        the pipeline: 7 stages, reproducible with `dvc repro`
├── tests/          49 tests incl. differential tests against a naive engine
└── scripts/arena.py   bot-vs-bot tournaments and an engine speed check
```

## Quick start

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -e .[game,dev]
```
```bash
.venv\Scripts\python -m barahai.game
```

Run the tests with `.venv\Scripts\python -m pytest`, and bot tournaments with
`.venv\Scripts\python scripts\arena.py random greedy alphabeta-d2 alphabeta-d4`.

---

## Phase 1: the MLOps pipeline

### The dataset problem, and how it is solved

Reinforcement learning has no dataset to download: the agent learns by playing. To get
versioned raw and processed data, and to make training faster, the project **generates
its own expert dataset**. Two alpha-beta searchers (depth 3) play 10,000 games, and every
move they choose is recorded. The network first learns to copy those moves (supervised
pretraining), then improves beyond them by playing with Maskable PPO. This is the same
recipe AlphaGo used: learn from expert games, then improve through play.

| Requirement | Implementation |
|---|---|
| Raw dataset | `data/raw/games.jsonl`: 10,000 expert games, 583,601 moves |
| Processed dataset | `data/processed/{train,val}.npz`: 392,148 positions, split **by game** (9,000 / 1,000) so no game leaks into validation |
| Test set | `data/puzzles.jsonl`: 291 positions from validation games where a depth-4 search finds one clearly best move |
| Versioning | DVC tracks all three datasets and the models; storage on AWS S3 (`dvc push` / `dvc pull`) |
| Preprocessing ("tokenization") | Each board becomes a 106-number vector from the mover's side; each of the 201 possible atomic moves has a fixed id, like a vocabulary |
| Training | Behaviour cloning on the expert moves, then Maskable PPO against a mix of bots and a frozen copy of itself |
| Validation | Accuracy on held-out games, win/draw/loss against 4 opponents, puzzle accuracy |
| Repeatable runs | `dvc.yaml` + `params.yaml`: `dvc repro` reruns only the stages whose code, data or settings changed |
| Experiment tracking | MLflow logs parameters, per-epoch and per-rollout metrics, and artifacts for every stage |
| Model registry | MLflow Model Registry: the ONNX policy is registered as `barahai-policy`, tagged with git commit, data hash and scores; a promotion rule assigns the `champion` or `challenger` alias |

### Pipeline

```
generate ──► preprocess ──► pretrain ──► train ──► evaluate ──► register
   │                                                   ▲
   └──────► puzzles ───────────────────────────────────┘
```

| Stage | What it does | Output |
|---|---|---|
| `generate` | 10,000 alpha-beta games on 8 processes; 2-8 random opening moves and 3% random moves for variety (random moves are never used as labels) | `data/raw/games.jsonl` |
| `preprocess` | Encodes every expert decision with 2+ legal moves; label = expert move, value = final result for the mover | `data/processed/` |
| `puzzles` | Scores every move of 6,000 validation positions with a depth-4 search; keeps the decisive ones | `data/puzzles.jsonl` |
| `pretrain` | Supervised learning: cross-entropy on the masked policy + MSE on the value head | `models/pretrained.pt` |
| `train` | Maskable PPO from the pretrained weights, 3M steps, 16 environments in parallel processes; self-play snapshot refreshed every 250k steps | `models/policy.{zip,pt,onnx}` |
| `evaluate` | 100 games vs each opponent (both colours, seeded random openings) with the ONNX model, plus the puzzle suite | `reports/eval.json` |
| `register` | Logs the ONNX model to MLflow, registers a new version, tags it, applies the promotion rule | `reports/registry.json` |

### Results (version 1, `champion`)

| Opponent | Win | Draw | Loss | Score |
|---|--:|--:|--:|--:|
| Random | 99% | 1% | 0% | 0.995 |
| Greedy (rule-based) | 76% | 24% | 0% | 0.880 |
| Alpha-beta depth 2 | 27% | 68% | 5% | 0.610 |
| Alpha-beta depth 3 (never trained against) | 18% | 71% | 11% | 0.535 |

* Pretraining: 78.2% accuracy at predicting the expert's move on held-out games (41 s).
* Puzzle accuracy: 63.9% of 291 decisive positions.
* Training speed: 2,347 steps/s. Measured on 8 cores: 16 environments in subprocesses
  2,370 steps/s, 8 in subprocesses 1,954, 8 in one process 1,698.

Score = (wins + draws / 2) / games. Promotion gate: score against alpha-beta depth 2 plus
puzzle accuracy must beat the current champion.

### CI gate

Besides lint and tests, CI runs a `puzzle-gate` job: it `dvc pull`s the champion
(`models/policy.onnx`) and the puzzle suite, replays the puzzles and fails if accuracy
drops more than `gate.puzzle_tolerance` below the baseline. On pull requests the baseline is
the base branch's `reports/eval.json`, so a weaker model cannot be merged; on pushes it is
the accuracy recorded in the same commit, so a code change that breaks the champion fails.
Run it locally with `.venv\Scripts\python -m barahai.rl.gate`. CI needs the repository
secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (read access to the DVC bucket).

### Run it

```bash
.venv\Scripts\python -m pip install -e .[game,dev,rl]
```
```bash
.venv\Scripts\dvc pull
```
```bash
.venv\Scripts\dvc repro
```
```bash
.venv\Scripts\mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

`dvc pull` downloads the datasets and models from S3 (needs AWS access); without it,
`dvc repro` regenerates everything from scratch in about 30 minutes on 8 cores. Change any
value in `params.yaml` and `dvc repro` reruns only what depends on it. Compare runs with
`dvc exp run -S train.init=scratch` and `dvc metrics diff`, or in the MLflow UI at
http://127.0.0.1:5000. Play against the trained model with
`.venv\Scripts\python scripts\arena.py onnx:models/policy.onnx alphabeta-d3`.

---

## The rules

### Board

A square of 5 × 5 points: 16 small boxes, the two long diagonals, and a diamond
joining the midpoints of the sides (the Alquerque board). Diagonal lines only pass
through the **strong points**, where `row + col` is even. These have up to 8
neighbours. The other points (**weak points**) connect only orthogonally.

```
  5  o───·───o───·───o        o strong point (diagonals)
     │ ╲ │ ╱ │ ╲ │ ╱ │        · weak point (orthogonal only)
  4  ·───o───·───o───·
     │ ╱ │ ╲ │ ╱ │ ╲ │        24 beads, centre empty:
  3  o───·───o───·───o          Turquoise: ranks 4-5 + a3, b3
     │ ╲ │ ╱ │ ╲ │ ╱ │          Carnelian: ranks 1-2 + d3, e3
  2  ·───o───·───o───·
     │ ╱ │ ╲ │ ╱ │ ╲ │
  1  o───·───o───·───o
     a   b   c   d   e
```

### Play (the BarahAI standard ruleset)

1. **Setup.** Each player has 12 beads filling the two rows nearest them plus the
   two middle-row points on their right. Only the centre is empty.
2. **Turns.** Carnelian moves first, then turns alternate. On your turn you either
   *step* or *capture*.
3. **Step.** Move one bead along a line to an adjacent empty point, in any
   direction. There is no "forward only" rule and no promotion.
4. **Capture.** Jump over an adjacent enemy bead, along a straight line, onto the
   empty point directly beyond it. The jumped bead is removed.
5. **Chain capture.** If the jumping bead can capture again from where it lands, it
   keeps jumping in the same turn. It may change direction between jumps.
6. **Captures are mandatory.** If any capture is available you must capture, and a
   chain must be continued while it can be. When several captures are available
   you may choose any of them (the longest is not required).
7. **Winning.** You win by capturing every enemy bead, or when your opponent has no
   legal move on their turn (blocked).
8. **Draws.** The game is drawn after **40 consecutive turns without a capture**,
   when the **same position with the same side to move occurs 3 times**, or at a
   300-turn safety cap.

### Why these choices

The historical records agree on the board, the setup, single-step moves, jump
captures and the win by capturing everything. They are silent on the rest, and
modern sources disagree:

| Question | What the sources say | BarahAI default |
|---|---|---|
| Multi-jump in one turn? | Not mentioned in the 1919/1923 Punjab record; allowed by AlignIt, twelvebead.com and 12beadsgame.com | **Yes**, direction may change |
| Capture compulsory? | Mandatory in the Bead 12 app and 12beadsgame.com; optional in many house rules (AlignIt) | **Mandatory** |
| Blocked player? | AlignIt and 12beadsgame.com: a win for the opponent | **Loses** |
| Draws? | 12beadsgame.com: 1 bead each + 8 quiet moves; others: by agreement | **40 quiet turns / threefold repetition** |

Every choice is a flag in `barahai.engine.Rules`, so other variants need no code
changes:

```python
Rules(forced_capture=True, chain_capture=True, quiet_limit=40,
      repetition_limit=3, max_turns=300, first_player=0)
```

With `forced_capture=False`, captures and chain continuations become optional, and
a `STOP` action ends a chain early.

### Notation

Files `a`-`e` run left to right and ranks `1`-`5` bottom to top, from Carnelian's
side. A step is `c2-c3` and a capture chain is `a1xc3xc1xa1`.

---

## Playing the game

| Input | Action |
|---|---|
| Click a bead, then a glowing point | Move (or drag the bead there) |
| Red ring on a point / around a bead | Capture landing spot / the bead you would take |
| Pulsing brass ring | A capture is available, so one of these beads must jump |
| `N` / `U` or `Ctrl+Z` / `H` | New game / take back your last turn / hint |
| `M` / `S` / `L` / `Esc` | Sound on-off / save / load (`~/.barahai/saved_game.json`) / deselect |

**Modes:** vs Computer (pick your colour), Two players (hotseat), Watch computers.

**Computer levels**

| Level | Engine | Notes |
|---|---|---|
| Naya (beginner) | Greedy, 30 % random moves | Grabs beads, avoids hanging its own, makes mistakes |
| Khiladi (player) | Alpha-beta, 3 turns | Deterministic for a given seed |
| Ustad (master) | Alpha-beta, iterative deepening, 1 s | As deep as the clock allows |

### The Chiniot board

The look is based on Chiniot (Punjab) sheesham-wood furniture with brass inlay:

- **Board.** Lacquered sheesham wood made from fractal noise, brass inlay lines in
  a routed groove, a band of brass diamonds with *khatim* (eight-pointed star)
  corner medallions, and a brass jali lattice behind it.
- **Beads.** Carnelian (*aqeeq*) and Persian turquoise (*feroza*). Each is shaded
  per pixel as a real dome: diffuse, specular and Fresnel lighting. Carnelian also
  transmits light, glowing from inside and casting orange light on the wood.
  Turquoise has a procedural spiderweb matrix.
- **Motion.** Steps ease out; jumps arc, lifting the bead and stretching its
  shadow. A captured bead bursts into sparks and flies to its captor's tray. The
  last turn stays traced on the board, and the winner's beads pulse in a wave.
- **Sound.** Synthesised with numpy: wood taps, stone clacks, and plucked-string
  phrases in Raag Yaman (Karplus-Strong) for a win, loss or draw.

Everything is baked once per window size and cached, so a frame is mostly blits:
about 3.3 ms against the 16.7 ms budget for 60 fps. Searches run in a separate
**process**. On a thread they froze the window for up to 1 s, because every
pygame blit gives up the GIL and then waits behind the busy search to get it back.
With the worker process the window holds 60 fps (worst frame 20 ms) while Ustad
thinks.

---

## Engine design

- **Bitboards.** A position is two 25-bit Python ints. Move generation shifts whole
  bitboards per direction against precomputed masks: about 116k actions/s in pure
  Python, single-threaded.
- **Atomic actions.** A turn is one or more actions. Each action is one step or one
  jump, encoded as `direction × 25 + from_square` (0-199), plus `STOP` = 200. Step
  and jump never share an id, because the neighbour in that direction is either
  empty or an enemy. Mid-chain, the same player simply acts again with only the
  jumping bead allowed to move. The action space is fixed at 201 ids.
- **Immutable states.** `GameState.apply(action)` returns a new state. Undo is
  popping a list, search needs no unmake logic, and states can be sent to the
  search process as they are.
- **Rules are enforced in the state.** `legal_actions()`, `action_mask()`,
  `is_legal()` and `describe()` (src, dst, captured) are the only ways to move.
  Illegal input raises `IllegalMoveError` and is never silently ignored.
- **Game records.** `GameRecord` stores the rules plus the action list as JSON and
  replays it exactly (it also validates every move on load).

### How it is verified

- **Differential testing.** [`tests/reference.py`](tests/reference.py) is a
  deliberately naive 2D-grid move generator that shares no code with the engine.
  The two must agree on every legal move in over 5,000 positions from random games,
  under each of the 4 capture-rule variants.
- **Symmetry.** The rules ignore orientation, and the strong-point pattern survives
  all 8 rotations and reflections of the square. A test checks that transforming a
  position transforms its legal moves the same way.
- **Perft.** Leaf counts from the start position at depths 1-6 are pinned:
  `4, 5, 6, 12, 19, 64`. The opening is narrow because the first move must enter
  the centre, and the reply is then a forced capture.
- **Rule scenarios.** Corner-turning chains, optional `STOP`, weak points having no
  diagonals, blocked-loses, threefold repetition, quiet-turn draws, bead
  conservation, and save/load round trips.
- **Strength ladder.** 20 games per pairing, colours alternated
  (`scripts/arena.py`):

| Pairing | Score | Draws | Avg turns |
|---|:-:|--:|--:|
| random vs greedy | 0 – 20 | 0 | 35 |
| random vs alphabeta-d2 | 0 – 20 | 0 | 28 |
| random vs alphabeta-d4 | 0 – 20 | 0 | 27 |
| greedy vs alphabeta-d2 | 1 – 9 | 10 | 64 |
| greedy vs alphabeta-d4 | 0 – 11 | 9 | 60 |
| alphabeta-d2 vs alphabeta-d4 | 0 – 14 | 6 | 54 |

The ladder is ordered: each stronger bot clearly beats the one below it. Draws
show up mostly between closer opponents, where neither side can force captures.

---

## Autopsy of the legacy C++ version

Line numbers refer to [`twelve_beads.cpp`](https://github.com/SilverUmer21/twelve_beads/blob/main/twelve_beads.cpp).

| # | Bug | Where | Effect |
|---|---|---|---|
| 1 | The turn switches whether or not a move happened | 595, 825 | Clicking a wrong point forfeits your turn and the bot moves |
| 2 | Clicking another of your own beads counts as a move attempt | 558 | You cannot change your selection without losing the turn |
| 3 | Out-of-bounds reads and writes: `board[r-2]`, `board[r][c+2]` with no bounds checks | 107-245 | Undefined behaviour. `board[r][5]` is really `board[r+1][0]`, so beads can wrap around the edge to the next row |
| 4 | The clicked destination is ignored | 561-593 | Clicking two points away steps one; clicking next to an enemy jumps |
| 5 | No chain captures, no forced capture, no blocked or draw rules | — | A blocked position hangs the game forever |
| 6 | The bot only scans interior points (`1..3`) and misses all edge moves | 249-251 | It passes silently when only edge moves exist |
| 7 | The bot's bounds guards check the wrong index (`i>=2` where `j-2` is used) | 296 | Illegal captures across rows |
| 8 | A failed parity check inside the bot's `else if` chain skips all later options | 259-432 | Valid moves are never tried |
| 9 | The win screen calls `Sleep(30000)` on the UI thread, and its "float" animation is computed once | 465-478 | The window freezes with "Not responding" for 30 s |
| 10 | Buttons poll `isButtonPressed` every frame | 649, 869, 999 | One click saves dozens of times; clicks fall through to the next screen |
| 11 | Hover plays the click sound every frame | 665, 958 | Sound spam |
| 12 | The background texture is reloaded from disk every frame | 1173 | Constant disk IO in the menu |
| 13 | Game flags are never reset after a game | 1190-1201 | Closing a game relaunches it forever; there is no way back to the menu |
| 14 | The board is a global that is never reset | 25 | A new game starts from the finished position |
| 15 | A 500 × 500 window holding a 25 px offset plus five 100 px cells | 489, 620 | The last column draws off-screen |
| 16 | Two copy-pasted 200-line game loops; Windows-only `Sleep`, `windows.h` | 484-927 | Fixes have to be made twice; it does not build elsewhere |
| 17 | `std::thread` started only to call `music.play()` (SFML already streams on its own thread) | 1192 | Pointless thread |

---

## Sources

- H. C. Das-Gupta (1923), "Notes on a Type of Sedentary Game Prevalent in Many Parts of India", *J. Proc. Asiatic Society of Bengal* 19: 71-74. Punjab and Simla, 1919, via [Ludii: Bára Guti](https://ludii.games/data.php?gameId=612)
- C. Das Gupta (1935), "A Few Types of Sedentary Games from Bihar", *J. Asiatic Society of Bengal* 1: 409-418. The circular Bihar variant, via [Ludii: Bara Guti (Bihar)](https://www.ludii.games/data.php?gameId=778)
- [gamerules.com: 12 Beads](https://gamerules.com/rules/12-beads/)
- [AlignIt Games: Bara Tehni rules](https://alignitgames.com/games/bara-tehni.html)
- [12beadsgame.com](https://12beadsgame.com/)
