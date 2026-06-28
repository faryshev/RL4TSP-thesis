#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# -----------------------------------------------------------------------------
# Launch preset.
#
# Leave exactly one block active. The environment variable RL4TSP_SCALE still
# overrides the active block, so these commands also work:
#   RL4TSP_SCALE=small bash run_experiments.sh
#   RL4TSP_SCALE=full  bash run_experiments.sh

# --- Small smoke run: quick correctness check before sending/rerunning.
SCALE="${RL4TSP_SCALE:-small}"

# --- Full thesis run: uncomment this block and comment the small block above.
# SCALE="${RL4TSP_SCALE:-full}"
# -----------------------------------------------------------------------------

if [[ "${SCALE}" != "small" && "${SCALE}" != "full" ]]; then
  echo "RL4TSP_SCALE must be 'small' or 'full'." >&2
  exit 1
fi

RUN_ID="${RL4TSP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RL4TSP_RUN_ROOT:-results/${RUN_ID}}"
CONFIG_DIR="${RUN_ROOT}/configs"
if [[ -e "${RUN_ROOT}" && "${RL4TSP_ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing run folder: ${SCRIPT_DIR}/${RUN_ROOT}" >&2
  echo "Choose a new RL4TSP_RUN_ID or set RL4TSP_ALLOW_OVERWRITE=1." >&2
  exit 1
fi
mkdir -p "${CONFIG_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/src"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${RL4TSP_MPLCONFIGDIR:-${SCRIPT_DIR}/.runtime-cache/matplotlib}"
export RL4TSP_TORCH_THREADS="${RL4TSP_TORCH_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${RL4TSP_TORCH_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${RL4TSP_TORCH_THREADS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${RL4TSP_TORCH_THREADS}}"

python3 - "${SCALE}" "${RUN_ROOT}" "${CONFIG_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

scale = sys.argv[1]
run_root = Path(sys.argv[2])
config_dir = Path(sys.argv[3])
config_dir.mkdir(parents=True, exist_ok=True)
suffix = "small" if scale == "small" else "full"
output_root = run_root / scale


def read_config(name: str) -> dict:
    return json.loads((Path("configs") / name).read_text(encoding="utf-8"))


def write_config(name: str, config: dict) -> None:
    (config_dir / name).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def reinforce_checkpoint_entries(root: Path) -> list[dict[str, str]]:
    return [
        {
            "method": "Attention:batch_greedy_mean",
            "model": "attention",
            "decoder": "greedy",
            "checkpoint": str(root / "models" / "attention_batch_greedy_mean.pt"),
        },
        {
            "method": "Attention:greedy_rollout",
            "model": "attention",
            "decoder": "greedy",
            "checkpoint": str(root / "models" / "attention_greedy_rollout.pt"),
        },
        {
            "method": "PointerNet:batch_greedy_mean",
            "model": "pointer",
            "decoder": "greedy",
            "checkpoint": str(root / "models" / "pointer_batch_greedy_mean.pt"),
        },
        {
            "method": "PointerNet:greedy_rollout",
            "model": "pointer",
            "decoder": "greedy",
            "checkpoint": str(root / "models" / "pointer_greedy_rollout.pt"),
        },
    ]


def pomo_checkpoint_entry(root: Path) -> dict[str, str | None]:
    return {
        "method": "POMO",
        "model": "attention",
        "decoder": "pomo",
        "checkpoint": str(root / "models" / "attention_pomo.pt"),
        "pomo_num_starts": None,
    }


reinforce = read_config(f"tsp_reinforce_{suffix}.json")
reinforce["output_root"] = str(output_root)
write_config("tsp_reinforce.json", reinforce)

pomo = read_config(f"tsp_pomo_{suffix}.json")
pomo["output_root"] = str(output_root)
write_config("tsp_pomo.json", pomo)

compare = read_config(f"tsp_compare_{suffix}.json")
compare["output_root"] = str(output_root)
compare["model_checkpoints"] = reinforce_checkpoint_entries(output_root)
compare["pomo_checkpoint"] = str(output_root / "models" / "attention_pomo.pt")
write_config("tsp_compare.json", compare)

diagnostics = read_config(f"tsp_diagnostics_{suffix}.json")
diagnostics["output_root"] = str(output_root)
all_entries = reinforce_checkpoint_entries(output_root) + [pomo_checkpoint_entry(output_root)]
diagnostics["model_checkpoints"] = all_entries
diagnostics["entropy_model_checkpoints"] = all_entries
write_config("tsp_diagnostics.json", diagnostics)

reward_noise = read_config(f"tsp_reward_noise_{suffix}.json")
reward_noise["output_root"] = str(output_root)
write_config("tsp_reward_noise.json", reward_noise)
PY

echo "Run id: ${RUN_ID}"
echo "Scale: ${SCALE}"
echo "Writing outputs under: ${SCRIPT_DIR}/${RUN_ROOT}"
echo "Runtime configs under: ${SCRIPT_DIR}/${CONFIG_DIR}"
echo "Matplotlib cache: ${MPLCONFIGDIR}"
echo "Torch/OpenMP threads: ${RL4TSP_TORCH_THREADS}"

FIGURES_DIR="${RUN_ROOT}/${SCALE}/figures"
TSP_COMPARISON_SUMMARY="${RUN_ROOT}/${SCALE}/tsp_comparison/summary.csv"

run_step() {
  local label="$1"
  shift
  local start
  local end
  start="$(date +%s)"
  echo
  echo "==== START ${label} ===="
  "$@"
  end="$(date +%s)"
  echo "==== DONE  ${label} ($((end - start))s) ===="
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo >&2
    echo "Missing expected ${label}: ${SCRIPT_DIR}/${path}" >&2
    echo "The previous step finished without creating the file required by the next step." >&2
    echo "Files currently present under ${SCRIPT_DIR}/${RUN_ROOT}:" >&2
    find "${RUN_ROOT}" -maxdepth 5 -type f -print >&2 || true
    exit 1
  fi
}

run_step "REINFORCE training" python3 -u experiments/run_tsp_reinforce.py --config "${CONFIG_DIR}/tsp_reinforce.json"
run_step "POMO training" python3 -u experiments/run_tsp_pomo.py --config "${CONFIG_DIR}/tsp_pomo.json"
run_step "scaling comparison" python3 -u experiments/run_tsp_compare.py --config "${CONFIG_DIR}/tsp_compare.json"
require_file "scaling summary CSV" "${TSP_COMPARISON_SUMMARY}"
run_step "scaling plots" python3 -u experiments/plot_tsp_summary.py --summary "${TSP_COMPARISON_SUMMARY}" --output-dir "${FIGURES_DIR}"
run_step "permutation, coordinate-noise, entropy diagnostics" python3 -u experiments/run_tsp_diagnostics.py --config "${CONFIG_DIR}/tsp_diagnostics.json"
run_step "reward-noise training" python3 -u experiments/run_tsp_reward_noise.py --config "${CONFIG_DIR}/tsp_reward_noise.json"
