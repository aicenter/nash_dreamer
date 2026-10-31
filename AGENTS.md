
## What this is

NashDreamer: a multi-agent DreamerV3 world model for two-player zero-sum imperfect-information games,
plus model-free baselines (RNaD, MMD) and a PPO best-response learner. An experimental decentralized
variant of the world model (one RSSM per player, RNaD only) also exists. Everything is JAX + Flax NNX;
games, losses and trajectory sampling are all jitted and vmapped.

## Repository layout

Python source lives under `src/`, one flat package per directory: modules resolve as `train.foo`,
`eval.foo`, `envs.foo`, not `src.train.foo`.

| Package | Contents |
|---|---|
| `src/nash_dreamer/` | The learners: `dreamer_ma.py`, `ma_rssm.py`, `networks.py`, `train_utils.py`, `replay_buffer.py`, `sim_rnad.py`/`sim_mmd.py`/`sim_ppo.py`, `rnad_dreamer.py`/`mmd_dreamer.py`, `optimizer.py`, `distributions.py`, and every `*_decentralized.py` sibling |
| `src/envs/` | `JaxGame` and every game implementation, plus `model_game.py` |
| `src/train/` | Per-game entry points and the shared training loop |
| `src/eval/` | Evaluation scripts, plus `pttt_exploitability/` (see its own [AGENTS.md](src/eval/pttt_exploitability/AGENTS.md)) |
| `src/plotting/` | `plot_metrics.py` and the world-model plot scripts; `plot_utils.py` is their shared helper |
| `src/world_model_experiments/` | The three world-model quality checks (see Evaluation) |
| `src/tabular_experiments/` | Scratch |

At the repo root: all `*.sh` wrapper scripts, `game_instances/`, and the generated-output directories
(`trained_networks/`, `metrics/`, `precomputed_metrics/`, `world_model_metrics/`, `world_model_plots/`,
`plots/`).

## Commands

### Install

Dependencies are managed by `uv` (`pyproject.toml` + `uv.lock`); `requirements.txt` is a legacy mirror,
not the source of truth.

```bash
uv sync                        # creates/updates .venv from uv.lock
uv run python -m train.goofspiel_train --num_cards 3 rnad --num_steps 1000 --seeds "(42,)"
```

**Use `uv run` rather than a bare `python`**, which may silently run against another project's
activated venv. `uv run` always uses this project's locked environment; its
`` `VIRTUAL_ENV=...` does not match the project environment path `` warning is expected. Every root
`*.sh` script already uses `uv run`.

Python `>=3.12.1` (`.python-version` pins the same). JAX is built for CUDA 12 unconditionally; this is a
GPU-only dependency set.

Constraints for future `uv lock` runs:
- TensorFlow and its ecosystem are deliberately absent, since nothing imports them. Viewing
  `jax.profiler` traces needs `xprof`; add it if profiling is ever used.
- `orbax-checkpoint` is pinned although nothing imports it (checkpoints are plain `pickle`, see
  `nash_dreamer.train_utils.save_model`/`load_model`): `flax` depends on it unconditionally, and the pin
  keeps its transitive dependencies from drifting on every re-lock.

If `uv sync` fails on a missing module, reinstall that one package rather than restoring TensorFlow.

### Running code

**Always run from the repo root with `src/` on `PYTHONPATH`.** `train.joint_train` builds checkpoint
paths from `os.getcwd()`. Every root `*.sh` script exports `PYTHONPATH="$(pwd)/src:$PYTHONPATH"`; do
the same ad hoc (`PYTHONPATH=src uv run python -m ...`).

```bash
# Training, called directly. First positional = algorithm; nash_dreamer takes a
# SECOND positional (reinforce|rnad|mmd) after all its flags.
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 nash_dreamer \
  --num_steps 1000 --seeds "(42,)" --encoded_classes 1 --encoded_categories 3 rnad --eta 0.2
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 rnad --num_steps 1000 --seeds "(42,)"

# PPO best response against a frozen opponent trained with --use_original_infoset.
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 ppo \
  --opponent_path trained_networks/rnad/goofspiel_3/seed_42/step_1000.pkl \
  --player_id 0 --num_steps 1000 --seeds "(42,)"

# Training via the shell wrappers (see the --clean_dir hazard below)
GAME=leduc ALGO=reinforce ./nash_dreamer_train.sh
GAME=goofspiel EXPERIMENT_ADD_FLAGS="--continue_train" ./nash_dreamer_train.sh

# Decentralized world model: no shell wrapper, only wired up for perturbed_rps.
PYTHONPATH=src uv run python -m train.perturbed_rps_train nash_dreamer_decentralized --use_original_infoset \
  --num_steps 1000 --seeds "(42,)" rnad --eta 0.2

# Evaluation and plots
PYTHONPATH=src uv run python -m eval.actor_critic_evaluate --game_name goofspiel_3 --seeds "(42,)" \
  --restore_step 10000 loaded --metric nash_conv --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad"
PYTHONPATH=src uv run python -m eval.head_to_head_evaluate --game_name goofspiel_4 ...
PYTHONPATH=src uv run python -m plotting.plot_metrics

# Budgeted approximate exploitability (metrics only, no checkpoints). See Evaluation.
ALGO_DIR=mmd GAME_NAME=goofspiel_3 RESTORE_STEP=1000 ./ppo_exploitability.sh

# World model quality checks against a trained DreamerMA checkpoint. See Evaluation.
./chance_marginal_evaluate.sh
./posterior_collapse_evaluate.sh
./world_model_sampling_evaluate.sh
```

There is no build, no linter, no test suite and no CI config.

### `--clean_dir` is destructive and on by default in the shell scripts

All four `*_train.sh` scripts default `EXPERIMENT_ADD_FLAGS="--clean_dir --save_first"`, and
`--clean_dir` does `rmtree` on `trained_networks/<algo>/<game>/seed_<N>/` before training (swallowing
any exception, so a failed delete only prints). Call the module directly for smoke tests, and never
pass `--clean_dir` over a directory whose contents matter. `ppo_exploitability.sh` never touches
`trained_networks/`.

## Architecture

### NashDreamer, RNaD, MMD and PPO, one training loop

[src/train/joint_train.py](src/train/joint_train.py)`::train_loop` is shared by all of them and is the
only place that handles checkpointing, seeds and `--continue_train`. It uses the game only for
`game.to_compact_str()` as the directory name.

- **NashDreamer** (`DreamerMA`) = an `MARSSM` world model + an actor-critic head, chosen by config type
  in `nash_dreamer/dreamer_ma.py::init`: `MMDConfig` → `MMDDreamer`, else `use_rnad` → `RNaDDreamer`,
  else `DreamerActorCritic` (REINFORCE). One `train_step` = sample the buffer, update the world model,
  then step the actor-critic on both real and imagined trajectories (`--beta_real` /
  `--beta_imagination`).
- **`SimRNaD`** / **`SimMMD`** are standalone baselines: no world model, the actor reads the real infoset
  tensor. Each shares its config with its Dreamer counterpart (`RNaDConfig`, `MMDConfig`), which
  ignores the Dreamer-only fields. MMD is Magnetic Mirror Descent with a uniform magnet: a PPO-style
  clipped surrogate plus a proximal-KL and a magnet-KL term. Loss math lives in the standalone learner
  and is imported by the Dreamer integration (`mmd_dreamer.py` ← `sim_mmd.py`,
  `rnad_dreamer.py` ← `sim_rnad.py`).
- **`SimPPO`** learns a best response to one frozen opponent loaded from `--opponent_path` (a
  `DreamerMA`/`SimRNaD`/`SimMMD` checkpoint trained with `--use_original_infoset`), not a policy for the
  game. It is strictly on-policy, with no replay buffer. `PPOConfig.entropy_coeff` defaults to 0 because
  entropy regularization softens the best response and under-estimates exploitability. Run standalone
  via `train.<game>_train ppo` (checkpointed under `trained_networks/ppo_p{player_id}/...`) or in memory
  via `eval/ppo_exploitability.py`.

### Decentralized world model (`nash_dreamer/*_decentralized.py`)

`DecentralizedMARSSM` gives each player its own RSSM over its own observation and action (weights shared
via `nnx.vmap` over the player axis), instead of `MARSSM`'s single recurrent state over the joint
observation. It uses real infosets only (asserts `--use_original_infoset`), reconciles two reward heads
as `(r0 - r1) / 2`, and its critic reads one player's own infoset. It is expected to be unstable: treat
it as an ablation target, not an alternative to `MARSSM`.

Only RNaD is implemented. `DecentralizedWMReplayBuffer` implements only the arrival-reward convention and
inherits `check_reward_alignment_once`. No shell or evaluation script supports these classes. The
world-model update and v-trace logic in `rnad_dreamer_decentralized.py` are duplicated from the
centralized versions, so bugfixes in `dreamer_ma.py` / `rnad_dreamer.py` must be ported by hand.

The only game wired to it is `JaxPerturbedRPS` (`src/envs/jax_rps_perturbed.py`). Plain `JaxRPS` is
useless for measuring convergence, because a zero-initialized policy head already is its uniform
equilibrium. The perturbed payoffs (Scissors beats Paper 2:0) keep the game value at 0 but move the
unique equilibrium to `(1/2, 1/4, 1/4)`.

### Latent vs. real infosets

By default `MARSSM` learns a latent infoset per player and the actor-critic runs on that;
`--use_original_infoset` switches to the real infoset and asserts
`information_state_tensor_shape() == observation_tensor_shape()`. `src/envs/model_game.py` supports both
in evaluation: `DreamerModelGame` exposes the learned model as a `JaxGame` (latent stochastic state
becomes chance outcomes), and `InformedRealGame` walks the real game while carrying latent infosets.
Evaluation scripts choose between them based on `use_real_infoset`.

### Replay buffer scans fixed-length trajectories

`nash_dreamer/replay_buffer.py` unrolls exactly `max_trajectory_length()` steps with `nnx.scan`
regardless of when the game ends, then filters chance nodes down to `max_trajectory_lenght_no_chance()`.
Every game must therefore keep **terminal absorbing** (sticky flag, zero reward afterwards) and **the
legal mask never all-zero**, since an action is sampled over the mask on every step, including past the
end of the game.

### Two different reward conventions, one per buffer

- `WMReplayBuffer` stores the reward for **arriving** at a step, filtered with the same `non_chance`
  mask as `obs`. `train_utils.wm_timestep_to_timestep` shifts it (`reward[1:]`) to the actor-critic
  convention.
- `ActorReplayBuffer` and `sim_ppo.py` store the reward for **acting** at a step, so each decision step
  must absorb the rewards of the chance steps that follow it. `replay_buffer.filter_chance_rewards` does
  this as a segment sum, independent of how chance and play nodes interleave.

`goofspiel_random_*` results from before 2026-07-31 are invalid (misaligned actor rewards).

`replay_buffer.check_reward_alignment_once` runs at every learner construction and raises if filtering
loses reward, which also catches an under-declared `max_trajectory_length()` or
`max_trajectory_lenght_no_chance()`. Do not remove it: this failure is otherwise silent.

### `legal_policy` shifts by the maximum over LEGAL actions

`_legal_shifted_logit` (`nash_dreamer/train_utils.py`), used by every learner through `ActorNetwork`,
shifts logits by the max over legal actions only. Illegal logits get no gradient and can grow without
bound; shifting by the global max would then underflow every legal probability in float32, giving NaN
gradients or an all-zero policy. `SimPPO.step` additionally raises on non-finite losses or parameters.

## The game layer (`src/envs/`)

[src/envs/jax_game.py](src/envs/jax_game.py) defines `JaxGame` with 13 abstract methods. Read
`jax_goofspiel.py::JaxGoofspiel` for the minimal deterministic case, `jax_leduc.py` for chance nodes
and turn-based play, `jax_rps.py` for the smallest possible example.

Contract details that are not visible from the abstract signatures:

- `get_info(state)` returns `(state_tensor, p1_infoset, p2_infoset, public_state)`, each a flat 1-D
  float vector of exactly the declared shape. Convention: prefix each player tensor with a player-id
  one-hot; zero every tensor at chance nodes.
- `apply_action(state, actions)` returns `(new_state, terminal, reward, legals)` where `legals` is
  `[num_players, num_distinct_actions]`, `terminal` is a scalar bool, and `reward` is a **scalar
  float32 from player 1's perspective** (p2's is its negation). Normalize into `[-1, 1]`.
- **Turn-based games** reserve action id `0` as a dummy `INVALID`, give the non-acting player exactly
  that one-hot as their legal mask, and index all effects by the acting player. `one_hot(action - 1, ...)`
  makes the dummy encode as all zeros. See `jax_leduc.py` and `jax_phantom_ttt.py`.
- **Deterministic games override none of the chance methods**; the `jax_game.py` defaults are correct.
  `apply_action` is still traced with the dummy joint action (the buffer's `lax.cond` traces both
  branches), so guard against that changing state.
- State is a `@chex.dataclass(frozen=True)` PyTree of fixed-shape arrays; methods carry
  `@functools.partial(jax.jit, static_argnums=(0,))` and must survive being vmapped over both the state
  axis and the action axis (see the double vmap in `eval/policy_eval_utils.py`).

### Adding a new game

There is **no game registry**; the shell scripts dispatch by module name
(`uv run python -m train."$GAME"_train`). A new game is two files:

1. `src/envs/jax_<name>.py` implementing `JaxGame`.
2. `src/train/<name>_train.py` — copy [src/train/goofspiel_train.py](src/train/goofspiel_train.py),
   add game-specific flags, dispatch on `args.experiment_type`. Game flags must come *before* the
   `nash_dreamer|rnad|mmd|ppo` positional.

`to_compact_str()` = `game_name()` + `params_dict()` **values** joined by `_`. It is both the checkpoint
directory name and the `--game_name` every evaluate/plot script takes, so changing `params_dict` silently
relocates checkpoints and breaks evaluation paths.

## Evaluation

[src/eval/policy_eval_utils.py](src/eval/policy_eval_utils.py) computes `nash_conv` / best responses by
an **exhaustive tree walk** that expands every history at a depth by all `A²` joint actions at once. It
is only viable for small games (Goofspiel-3, Leduc, RPS) and runs out of memory on larger ones. It also
assumes both players have the same number of infosets at every depth, and otherwise raises a `ValueError`
(inhomogeneous shape) in `extract_model_policy`, as it does on Phantom TTT. Latent infosets can violate this
too, since they are grouped by a distance tolerance. For such games, use `eval/head_to_head_evaluate.py` for an
empirical win rate, or one of the two exploitability tools below.

`src/eval/goofspiel_nash.pkl` and `leduc_nash.pkl` are reference equilibria for the `nash` subcommand.
Everything under `metrics/`, `precomputed_metrics/`, `world_model_metrics/` is generated output.

### Exact exploitability for Phantom TTT (`src/eval/pttt_exploitability/`)

Exports the policy to `eas` ("exp-a-spiel"), an external C++ treeplex solver that is **not a pinned
dependency** and must be built separately, and computes an exact NashConv, as defined in
`policy_eval_utils.nash_conv`. Driven end-to-end by the root `pttt_exploitability.sh`. Real-infoset
actors only; a latent-infoset `MARSSM` is rejected.

See [src/eval/pttt_exploitability/AGENTS.md](src/eval/pttt_exploitability/AGENTS.md) before touching
anything in that directory or the root `pttt_exploitability.sh` — in particular, that script
deliberately carries no cluster preamble, and must not gain one.

### Budgeted exploitability (`src/eval/ppo_exploitability.py`)

Trains `SimPPO` best responses in memory and plays each head-to-head against the frozen opponent,
writing metrics only. It sweeps the seed directories of one algorithm at a fixed `--restore_step` (or
takes a single `--opponent_path`) and reports `max_seed BR(p0) + max_seed BR(p1)`.

The PPO budget is part of the metric: an unbounded budget saturates, scoring early and converged
checkpoints of a hard game alike. The result is therefore a budgeted lower bound. Never compare it with
the exact `pttt_exploitability` numbers (hence the output name `approx_exploitability.tsv`), and keep
`--num_steps` / `--ppo_seeds` / `--batch_size` / `--num_epochs` identical across everything compared
(each row records them; the script warns on mismatch). A negative `expl_p` means the search did not reach
the profile's own value within the budget; it is reported, not clamped.

### World model quality (`src/world_model_experiments/`)

Three checks against a trained `DreamerMA` checkpoint, each with a root-level `<name>_evaluate.sh`
wrapper. All walk the tree under the model's own policy, pruned by `--policy_eps`, and accept
`--restore_step -1` to sweep every checkpoint.

- `chance_marginal_eval.py`: distance (L∞ / total variation) between the model's prior next-observation
  distribution and the real one, split by chance vs. deterministic nodes.
- `posterior_collapse_eval.py`: KL(posterior‖prior) at each chance node given the realized outcome, to
  check that the stochastic state encodes chance outcomes.
- `world_model_sampling_eval.py`: imagined rollouts run alongside the real game, flagging steps where
  obs/reward/terminal/legal diverge past a threshold.

## Writing comments

Keep comments and docstrings short: state what the code does and why, ideally in one sentence.
Do not put in them references to a Claude Code session, commit hashes, numbers measured during a
session, or history of how the code used to work — that belongs in commit messages. Don't
re-explain concepts documented elsewhere; name them. A comment describing buggy or non-conforming
behavior should be raised with the user rather than written up at length. The user can override
any of this for a specific comment. The `refactor-comments` skill (`.claude/skills/refactor-comments/`)
applies these rules to an existing file.

## Repo-specific traps

- `max_trajectory_lenght_no_chance` — the typo is part of the API, spelled that way everywhere.
- `src/eval/pttt_exploitability/` needs the external `eas` package; `import eas` fails after a bare
  `uv sync`.
- Per-game hyperparameters live in hardcoded bash `if [ "$GAME" == ... ]` blocks duplicated across the
  `*_train.sh` scripts; there are no config files.
- `--wm_warm_up` defaults to 1000, so short NashDreamer runs never exercise the imagination path. Set it
  to 1 when smoke-testing a new game.
