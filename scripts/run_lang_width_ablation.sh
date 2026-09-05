#!/usr/bin/env bash
# =============================================================================
# run_lang_width_ablation.sh — controlled language-context-width ablation
# =============================================================================
#
# Question: does widening the static ONNX language input (16 -> 24 -> 32 tokens)
# recover the closed-loop LIBERO-Spatial success lost after ONNX deployment?
#
# Arms (all FP32 monolithic ONNX, same tether 0.12.0 pipeline, same checkpoint):
#   w16 = exports/smolvla_libero          (existing paper export)
#   w24 = exports/smolvla_libero_lang24   (scripts/export_smolvla_lang_width.py)
#   w32 = exports/smolvla_libero_lang32
#
# Per arm we measure: CUDA-EP p50/p99 latency + peak memory (uniform bench),
# closed-loop Spatial and Object success (pinned lerobot-eval protocol:
# 100 eps/suite, seed 42, n_action_steps=1, 10 flow steps, batch 1).
#
# Runtime on RTX 2060: ~40 min per export, ~10 min per bench, ~3 h per suite.
# Everything logs to results/lang_ablation/. Safe to re-run: finished steps
# are skipped when their output already exists.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/lang_ablation"
mkdir -p "${OUT}"
MAIN_LOG="${OUT}/pipeline.log"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

log() { echo "==> $(date -Is) $*" | tee -a "${MAIN_LOG}"; }

export_dir() { case "$1" in 16) echo "${ROOT}/exports/smolvla_libero";; *) echo "${ROOT}/exports/smolvla_libero_lang$1";; esac; }

do_export() {  # width
  local w="$1" dir; dir="$(export_dir "$w")"
  if [[ -f "${dir}/model.onnx" && -f "${dir}/tether_config.json" ]]; then log "export w${w}: exists, skip"; return 0; fi
  log "export w${w}: start"
  conda activate vla-export
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/env.sh" >/dev/null
  export HF_HUB_DISABLE_XET=1
  python "${ROOT}/scripts/export_smolvla_lang_width.py" --lang-len "${w}" --output "${dir}" \
    >> "${OUT}/export_lang${w}.log" 2>&1
  local rc=$?
  log "export w${w}: done rc=${rc}"
  return $rc
}

wait_for_export() {  # width — wait for an export started outside this script
  local w="$1" dir; dir="$(export_dir "$w")"
  while pgrep -f "export_smolvla_lang_width.py --lang-len ${w} " >/dev/null 2>&1; do sleep 60; done
  [[ -f "${dir}/model.onnx" ]] || { log "export w${w}: model.onnx missing after wait"; return 1; }
  log "export w${w}: ready"
}

do_bench() {  # width
  local w="$1" dir json; dir="$(export_dir "$w")"; json="${OUT}/w${w}/latency.json"
  if [[ -f "${json}" ]]; then log "bench w${w}: exists, skip"; return 0; fi
  log "bench w${w}: start"
  conda activate vla-export
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/env.sh" >/dev/null
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/env_ort_cuda12.sh" >/dev/null
  mkdir -p "${OUT}/w${w}"
  python "${ROOT}/scripts/bench_onnx_lang_width.py" --export "${dir}" --out "${json}" \
    >> "${OUT}/w${w}/bench.log" 2>&1
  local rc=$?
  log "bench w${w}: done rc=${rc}"
  return $rc
}

do_eval() {  # width suite [task_ids n_episodes label]
  local w="$1" suite="$2" task_ids="${3:-}" n_eps="${4:-10}" label="${5:-}"
  local dir odir; dir="$(export_dir "$w")"
  odir="${OUT}/w${w}/${suite}${label}"
  if [[ -f "${odir}/eval_info.json" ]]; then log "eval w${w} ${suite}${label}: exists, skip"; return 0; fi
  log "eval w${w} ${suite}${label}: start (eps/task=${n_eps} tasks=${task_ids:-all})"
  conda activate vla
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/env.sh" >/dev/null
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/env_ort_cuda12.sh" >/dev/null
  mkdir -p "${odir}"
  local extra=()
  [[ -n "${task_ids}" ]] && extra+=(--task-ids "${task_ids}")
  python "${ROOT}/scripts/libero_onnx_eval.py" --export "${dir}" --tag "lang${w}" --suite "${suite}" \
    --n-episodes "${n_eps}" --seed 42 --output "${odir}" "${extra[@]}" >> "${odir}/eval.log" 2>&1
  local rc=$?
  log "eval w${w} ${suite}${label}: done rc=${rc} $(python -c "import json,sys;print(json.load(open('${odir}/eval_info.json'))['overall']['pc_success'])" 2>/dev/null)"
  return $rc
}

log "pipeline start (pid $$)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | tee -a "${MAIN_LOG}"

# 1. exports
wait_for_export 24 || do_export 24 || exit 1
do_eval 24 libero_spatial 0 2 _smoke_task0 || exit 1     # shape/wiring check before spending hours
do_export 32 || exit 1
do_eval 32 libero_spatial 0 2 _smoke_task0 || exit 1

# 2. uniform latency/memory bench for all three arms (quick)
do_bench 16; do_bench 24; do_bench 32

# 3. closed-loop: Spatial first (the informative suite), then Object
do_eval 24 libero_spatial
do_eval 32 libero_spatial
do_eval 24 libero_object
do_eval 32 libero_object

conda activate vla
python "${ROOT}/scripts/summarize_lang_ablation.py" >> "${MAIN_LOG}" 2>&1 || true
log "pipeline complete"
