# Shared LUMI paths and helpers — source from other scripts.
WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
PARTITION="${PARTITION:-small-g}"
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"

load_aitta_token() {
    local token_file="${AITTA_API_TOKEN_FILE:-${WORK_DIR}/conf/.aitta_token}"
    if [[ -z "${AITTA_API_TOKEN:-}" && -f "${token_file}" ]]; then
        export AITTA_API_TOKEN="$(tr -d '[:space:]' < "${token_file}")"
    fi
}

run_in_container() {
    module purge 2>/dev/null || true
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
    singularity exec \
        --bind /scratch/project_462001520:/scratch/project_462001520 \
        --bind /pfs/lustrep1:/pfs/lustrep1 \
        --pwd "${WORK_DIR}" \
        "${CONTAINER}" \
        bash -lc "source '${VENV_ACTIVATE}' && $*"
}

free_devg_slot() {
    local blocking
    blocking=$(squeue -u "${USER}" -h -o "%i %P %j" | awk '$2=="dev-g" && $3=="bash" {print $1; exit}')
    if [[ -n "${blocking}" ]]; then
        echo "Cancelling blocking interactive job ${blocking} on dev-g..." >&2
        scancel "${blocking}"
        sleep 3
    fi
    pkill -f "run_mini_in_allocation.sh" 2>/dev/null || true
}
