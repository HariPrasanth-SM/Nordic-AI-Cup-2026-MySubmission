# Survival Simulator — Submitted Solution

Nordic AI Cup 2026 · Submission by The Winter Soldier

## Submitted configuration

The submitted solution is a **stateful, hierarchical heuristic controller with short-horizon model-predictive predator avoidance and colony-level reproduction management**. It runs on CPU and requires no neural-network checkpoint or training to reproduce.

The exact configuration selected after local comparisons is:

```text
runs/submission_20260920_152628/configs/baseline.json
```

This was copied from `configs/fin4_baseline.json`. Its `controller` field is `fin3`. Although the server entry point is named `server6`, that module routes this configuration to the original `survivor.policy.Controller`; it does not activate PPO or QR-DQN.

Keep the submitted configuration and the accompanying source code together. The frozen submission copy is authoritative; the commands below assume the existing challenge repository with the fin4, fin5 and fin6 additions already installed.

## Files to submit and archive command

Submit the complete `survivor/` package, the exact frozen baseline configuration, the original simulator and requirements, and this README. Preserve the original challenge template files alongside the solution. The main behavior comes from `policy.py`, `colony.py` and `pursuit.py`, served through `server6.py`; the remaining package modules preserve its loader and evaluator dependencies.

| Path relative to `survival-simulator/` | Include |
| --- | --- |
| `survivor/` | Entire source package, excluding `__pycache__` and compiled `.pyc` files. |
| `src/` | Original challenge simulator and its support files. |
| `requirements.txt` | Dependency list. |
| `configs/fin4_baseline.json` | Original baseline configuration. |
| `runs/submission_20260920_152628/configs/baseline.json` | Exact frozen configuration used for submission. |
| `api.py`, `agent_server.py` | Original challenge files, unchanged, wherever present in the template. |
| `README_SUBMISSION.md` | This reproduction guide. |

Training logs, other experiment run folders, the virtual environment, and trained `.pt`/`.npz` checkpoints are not needed for this baseline. Download this README into the `survival-simulator/` directory before packaging.

Run this Bash command from that directory. It includes the original API files when present:

```bash
SUBMISSION_FILES=(
  survivor src requirements.txt
  configs/fin4_baseline.json
  runs/submission_20260920_152628/configs/baseline.json
  README_SUBMISSION.md
)
for TEMPLATE_FILE in api.py agent_server.py; do
  if [ -f "$TEMPLATE_FILE" ]; then
    SUBMISSION_FILES+=("$TEMPLATE_FILE")
  fi
done

tar --exclude='__pycache__' --exclude='*.pyc' \
  -czf survival_submission.tar.gz "${SUBMISSION_FILES[@]}"
```

Inspect the packaged file list:

```bash
tar -tzf survival_submission.tar.gz
```

On a fresh machine, extract into a new directory and follow the installation, server and local evaluation commands below:

```bash
mkdir -p survival-submission
tar -xzf survival_submission.tar.gz -C survival-submission
cd survival-submission
```

The archive contains the existing solution files; this README does not itself include their source code. No retraining is required.

## Setup and reproduction

Run commands from the `survival-simulator` directory, which contains `survivor/`, `configs/`, `src/` and `requirements.txt`.

Use the existing working environment:

```bash
source .simulator_env/bin/activate
```

For a fresh environment, use Python 3.12 and install the repository requirements:

```bash
python3.12 -m venv .simulator_env
source .simulator_env/bin/activate
python -m pip install -r requirements.txt
```

PyTorch and CUDA are unnecessary for the submitted baseline, including its local evaluation. They were used only for experimental training. Retain the complete `survivor/` package: the shared loader imports experimental modules even when their policies are inactive.

If the frozen submission directory was not copied to a new machine, reconstruct its configuration from the unchanged baseline file:

```bash
mkdir -p runs/submission_20260920_152628/configs
cp configs/fin4_baseline.json runs/submission_20260920_152628/configs/baseline.json
```

Do not overwrite an existing frozen configuration with a subsequently edited baseline. To check that the two copies still agree:

```bash
cmp configs/fin4_baseline.json runs/submission_20260920_152628/configs/baseline.json
```

No output and exit status zero indicate identical files.

## Run the submitted API server

Stop any previous agent server occupying port 9052, then run:

```bash
SURVIVOR_PARAMS=runs/submission_20260920_152628/configs/baseline.json \
PORT=9052 python -m survivor.server6
```

In a separate terminal:

```bash
curl http://127.0.0.1:9052/
curl http://127.0.0.1:9052/stats
```

The root response should identify `controller: fin3` and the selected configuration path. During evaluation, `/stats` exposes request counts, simulation time, score, controller errors and policy timing. The error counter should remain zero; a successful HTTP response alone does not establish that the controller ran without errors because the server has fallback actions.

For organizer validation, keep the server and the existing public tunnel running and provide the externally reachable endpoint in the competition interface:

```text
http://<YOUR_PUBLIC_HOST>:9052/predict
```

Use the public URL/port actually configured for the tunnel. The local status requests above do not start a scored organizer evaluation. No separate organizer evaluation-launch API command was established in this project.

The server handles `POST /predict` and returns `{"actions": [...]}`. Each action contains `agent_id`, `move_distance`, `move_direction`, `turn_angle` and `spawn_agent`. It loads the configuration at startup and rebuilds controller state when simulation time moves backwards for a new episode. Restart the server to change configurations. Keep it as a single process because controller memory is stateful. The challenge-provided `api.py` and `agent_server.py` do not need changes.

## Local evaluation

Reproduce the final 12-seed baseline check:

```bash
python -m survivor.fin6 evaluate \
  --configs runs/submission_20260920_152628/configs/baseline.json \
  --seeds 12 --workers 8 --minutes 10 \
  --seed 2502000 --horizon 3000 \
  --out "runs/submission_reproduce_$(date +%Y%m%d_%H%M%S)"
```

This evaluates seeds 2502000–2502011. For the two earlier 12-seed batches, repeat with `--seed 2500000` and `--seed 2501000`, using a fresh output directory each time. The configuration and controller seed are fixed; timings depend on the machine, and reproducing scores requires the same simulator and policy code.

`--minutes` is the total wall-clock budget for the evaluation batch, not an episode's simulated duration. `--horizon 3000` sets the maximum simulated duration. If the budget expires, unfinished episodes are censored and excluded from score summaries. Check that the summary reports the intended number of completed games; increase the time budget if necessary.

Outputs:

| Output | Contents |
| --- | --- |
| Terminal `game` events | Score, survival time, policy time and peak population per completed game. |
| Terminal `summary` events | Completed-game count, mean, median, worst-20% mean (`cvar20`), maximum score and mean policy time. `full` counts games reaching the requested horizon. |
| `games.jsonl` | Per-game records and diagnostics, including censored records where available. |
| `report.json` | Summary statistics and paired comparisons when several configurations are supplied. |
| `evaluated_configs.json` | Snapshot of configurations used in this run. |

The evaluator drives `src.core.SimulationCore` locally, using the payload structure expected by the controller and the `fastsim` acceleration helpers. It bypasses HTTP and the public tunnel. Its policy timing therefore excludes serialization, network latency and organizer-side overhead; use server validation to check end-to-end operation.

## Approach

1. **Observation and memory.** Maintain per-agent state across ticks, including remembered food/tree locations, recent threats and movement history. Decisions use the agent observations supplied through the API rather than hidden simulator state.
2. **Foraging and energy conservation.** Choose between collecting fruit, waiting near useful trees, resting and exploring. Fruit maturity, energy urgency, crowding and recent unsuccessful visits influence these decisions. Avoiding unnecessary movement preserves energy for future survival and escape.
3. **Predator avoidance.** When threatened, evaluate candidate directions, speed modes and facing choices with a lightweight pursuit model. The configured horizon is 24 ticks, approximately 2.4 simulated seconds. Execute the first action and replan from new observations. This is approximate model-predictive control, not a learned dynamics model.
4. **Colony management.** Coordinate births using population demand, parent energy reserves, age, nearby food and trait preferences. The population target decays over time, reducing resource pressure as conditions become less favorable. Birth spacing and young-cohort limits help avoid synchronized population growth; older eligible parents can transfer energy into offspring.

The goal is the simulator's score, dominated by colony survival time with fruit bonuses and predation penalties. A larger population is useful only when it improves continued survival; it is not itself the objective. The submitted configuration enables colony management and MPC escape planning, while disabling the optional learned escape and value-model gates.

## Code files and roles

| File or directory | Role |
| --- | --- |
| `runs/submission_20260920_152628/configs/baseline.json` | Frozen parameters selected for submission. |
| `configs/fin4_baseline.json` | Source baseline configuration from which the frozen copy was made. |
| `survivor/server6.py` | FastAPI entry point, configuration loading, episode reset, action-ID checks and runtime statistics. |
| `survivor/policy.py` | Active `Controller` and `Params`: observation memory, foraging, resting, exploration, threat handling and action generation. |
| `survivor/colony.py` | Active colony-level reproduction decisions, energy reserves, population/age management and diagnostics. |
| `survivor/pursuit.py` | Vectorized candidate escape rollouts used by MPC predator avoidance. |
| `survivor/colony_ppo.py` | Shared loader's first routing layer; PPO supervisor is inactive for `controller: fin3`. |
| `survivor/qr_escape.py` | Shared loader's second routing layer; learned escape override is inactive for this configuration. |
| `survivor/frontier.py` | Final routing layer that instantiates the original `Controller` for `fin3`; experimental frontier policy is inactive. |
| `survivor/escape_model.py`, `survivor/value_model.py` | Imported optional model utilities; their learned gates are disabled and their weights are not required. |
| `survivor/fin6.py`, `survivor/fin5.py`, `survivor/fin4.py` | Shared local evaluation entry point, scheduling/reporting and full-game execution; also contain experimental training/search commands. |
| `survivor/runner.py`, `survivor/fastsim.py` | Local action adapter and simulation acceleration support. |
| `src/` | Challenge simulator used for local evaluation. |
| `requirements.txt` | Runtime and simulator dependencies. |

The baseline server loading path is:

```text
server6 → colony_ppo.load_controller → qr_escape.load_controller
        → frontier.load_controller → policy.Controller
```

Retaining loader modules does not imply that their experimental neural policies are active.

## Selection evidence

Scores below are local results supplied by the participant, not organizer validation scores.

| Model | First 24 games: mean | First 24 games: median | Additional 12 games: mean | Combined 36 games: mean |
| --- | ---: | ---: | ---: | ---: |
| Submitted baseline | 1080.63 | 1079.51 | 1060.32 | approximately 1073.86 |
| fin4 challenger | 1050.39 | 1047.18 | 1042.10 | approximately 1047.63 |
| PPO greedy | 1026.18 | 1004.88 | Not run | — |
| QR-DQN mean | 991.96 | 1016.72 | Not run | — |

The baseline had the strongest mean and median in the 24-game comparison and retained a higher mean in the final 12-game check. The challenger's final paired difference was −18.22 points, with a 95% interval of [−215.62, 179.18]. This is not a statistically decisive separation, but there was no demonstrated gain supporting a switch. Neither baseline nor challenger reached 3000 seconds in the final check. Baseline policy computation averaged approximately 1.91 ms per tick in that batch.

The accumulated response-time budget reported for the competition was increased to 1200 seconds. At 10 ticks per simulated second, surviving 3000 seconds would require approximately 30,000 responses, leaving 40 ms per response on average. Local policy time covers only part of that budget.

## Other approaches explored

- **fin4 parameter search / challenger:** searched strategy and controller parameters with simulation-based comparisons. Some individual runs scored higher, but the final matched-seed averages did not justify replacing the baseline.
- **fin5 QR-DQN escape policy:** trained a distributional action-value network with 49 discrete escape choices and 32 return quantiles. It targeted predator encounters while preserving the surrounding controller. Fast encounter training did not translate into a reliable full-game score improvement.
- **fin6 PPO colony supervisor:** learned to select among eight high-level strategy configurations every 20 simulated seconds, using full-game score increments as reward. The greedy and sampled variants did not outperform the baseline in the available evaluations.

These experiments remain in the repository for reproducibility, but no experimental neural checkpoint is used by the submitted configuration.
