#!/bin/bash
# =============================================================================
#  run_full_sweep.sh  —  ONE COMMAND to run the full ADMET sweep on LUMI.
# =============================================================================
#
#  WHAT THIS DOES (plain English)
#  ------------------------------
#  It runs the whole experiment defined in conf/sweep.yaml:
#    1. Trains every (endpoint, arm, n, seed) combination as SLURM jobs on GPUs.
#    2. After they all finish, it collects the numbers, draws the plots, and
#       writes an AI-generated summary report.
#  When it is done you get, inside the  results/  folder:
#    results/results.parquet   <- all numbers in one table
#    results/curves.png        <- learning curves (error vs #training samples)
#    results/ma_rae.png        <- the headline aggregate metric
#    results/report.md         <- written characterization (read this first)
#
#  WHO CAN RUN IT
#  --------------
#  Anyone. You do NOT need to understand SLURM, Singularity, or the code.
#  Just follow the 3 steps below.
#
#  STEP-BY-STEP (for a human on the LUMI login node)
#  -------------------------------------------------
#    1. Get an Aitta token (needed for the AI summary). Open this in a browser:
#           https://aitta-auth.csc.fi/myToken
#       Copy the long token string.
#    2. Save it once (replace PASTE_TOKEN_HERE with your token):
#           echo 'PASTE_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token
#    3. Run this script from the project root:
#           bash scripts/run_full_sweep.sh
#       Answer "y" when it asks to confirm. That's it.
#
#  Then check progress any time with:
#           bash scripts/check_status.sh
#
#  OPTIONS
#  -------
#    --mini        Run the small 8-job test sweep (conf/sweep-mini.yaml) instead
#                  of the full 240-job sweep. Good for a quick check first.
#    --yes, -y     Don't ask for confirmation (used by automated agents).
#    --dry-run     Show what WOULD be submitted, but don't actually submit.
#    --token TOK   Provide the Aitta token directly on the command line.
#    --free-slot   Cancel a leftover interactive "bash" GPU job that may be
#                  blocking submission, then continue.
#    --help, -h    Show this help and exit.
#
#  EXAMPLES
#  --------
#    bash scripts/run_full_sweep.sh                 # full sweep, ask to confirm
#    bash scripts/run_full_sweep.sh --mini          # quick 8-job test
#    bash scripts/run_full_sweep.sh --yes           # full sweep, no prompt (agents)
#    bash scripts/run_full_sweep.sh --dry-run       # preview only
#    bash scripts/run_full_sweep.sh --token eyJ...  # pass token inline
# =============================================================================
set -euo pipefail

# --- Always run from the project root, whatever directory you're in ----------
cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

# --- Defaults ----------------------------------------------------------------
MODE_FULL=1          # full sweep by default
ASSUME_YES=0
DRY_RUN=0
FREE_SLOT=0

# --- Parse the options -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mini)        MODE_FULL=0 ;;
        --full)        MODE_FULL=1 ;;
        -y|--yes)      ASSUME_YES=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        --free-slot)   FREE_SLOT=1 ;;
        --token)       shift; export AITTA_API_TOKEN="${1:?--token needs a value}" ;;
        -h|--help)     sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

if [[ "${MODE_FULL}" -eq 1 ]]; then
    PLAN="conf/sweep.yaml"
    SWEEP_FLAGS=(--full)
    LABEL="FULL sweep (conf/sweep.yaml)"
else
    PLAN="conf/sweep-mini.yaml"
    SWEEP_FLAGS=()
    LABEL="MINI test sweep (conf/sweep-mini.yaml)"
fi

echo "============================================================"
echo " ADMET few-shot transfer — ${LABEL}"
echo "============================================================"

# --- Pre-flight check 1: are we on a machine that can submit jobs? -----------
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' was not found." >&2
    echo "       Run this on a LUMI LOGIN NODE (e.g. uan01/uan02/uan03)," >&2
    echo "       not inside a container or on your laptop." >&2
    exit 1
fi

# --- Pre-flight check 2: do we have the AI (Aitta) token? --------------------
load_aitta_token
if [[ -z "${AITTA_API_TOKEN:-}" ]]; then
    echo
    echo "MISSING: Aitta API token (needed for the AI-written report)." >&2
    echo "Fix it in ONE of these ways, then re-run this script:" >&2
    echo >&2
    echo "  A) Save it to a file (recommended):" >&2
    echo "       1. Open https://aitta-auth.csc.fi/myToken and copy your token" >&2
    echo "       2. echo 'PASTE_TOKEN_HERE' > conf/.aitta_token && chmod 600 conf/.aitta_token" >&2
    echo >&2
    echo "  B) Pass it on the command line:" >&2
    echo "       bash scripts/run_full_sweep.sh --token PASTE_TOKEN_HERE" >&2
    echo >&2
    echo "  C) Export it in your shell:" >&2
    echo "       export AITTA_API_TOKEN=PASTE_TOKEN_HERE" >&2
    exit 1
fi
echo "[ok] Aitta token found."

# --- Pre-flight check 3: does the plan file exist? ---------------------------
if [[ ! -f "${PLAN}" ]]; then
    echo "ERROR: plan file ${PLAN} not found." >&2
    exit 1
fi

# --- Show the user exactly what will happen ----------------------------------
echo "[ok] Plan file: ${PLAN}"
echo
echo "This plan will run these combinations:"
sed 's/^/      /' "${PLAN}"
echo

# --- Confirm (humans), auto-proceed for agents / --yes / non-interactive -----
if [[ "${ASSUME_YES}" -eq 0 && "${DRY_RUN}" -eq 0 && -t 0 ]]; then
    read -r -p "Submit these jobs to SLURM now? [y/N] " reply
    case "${reply}" in
        y|Y|yes|YES) ;;
        *) echo "Aborted. Nothing was submitted."; exit 0 ;;
    esac
fi

# --- Build the argument list for the real worker script ----------------------
ALL_ARGS=("${SWEEP_FLAGS[@]}")
[[ "${FREE_SLOT}" -eq 1 ]] && ALL_ARGS+=(--free-slot)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo
    echo "DRY-RUN: would submit with:  bash scripts/submit_all.sh ${ALL_ARGS[*]:-}"
    echo "         (rendering SLURM scripts only, not submitting)"
    # Render scripts without submitting so the user can inspect them.
    RESULTS_DIR_PREVIEW=$([[ "${MODE_FULL}" -eq 1 ]] && echo results || echo results-mini)
    JOB_NAME_PREVIEW=$([[ "${MODE_FULL}" -eq 1 ]] && echo admet-sweep || echo admet-mini)
    PLAN="${PLAN}" RESULTS_DIR="${RESULTS_DIR_PREVIEW}" JOB_NAME="${JOB_NAME_PREVIEW}" \
        bash scripts/submit_sweep.sh --llm --dry-run
    echo "Rendered: ${RESULTS_DIR_PREVIEW}/${JOB_NAME_PREVIEW}.sbatch"
    exit 0
fi

# --- Do it -------------------------------------------------------------------
echo
echo ">>> Submitting. This prints two SLURM job IDs (sweep, then collect)."
bash scripts/submit_all.sh "${ALL_ARGS[@]}"

# --- Tell the user what to do next -------------------------------------------
RESULTS_DIR=$([[ "${MODE_FULL}" -eq 1 ]] && echo results || echo results-mini)
echo
echo "============================================================"
echo " Submitted. You can close your laptop — SLURM runs it for you."
echo
echo " Check progress:        bash scripts/check_status.sh"
echo " When finished, read:   ${RESULTS_DIR}/report.md"
echo " Plots:                 ${RESULTS_DIR}/curves.png  ${RESULTS_DIR}/ma_rae.png"
echo "============================================================"
