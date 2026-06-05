from pathlib import Path

from experiment.slurm import SlurmConfig, render_sbatch, render_worker


def test_sbatch_renders_with_array_size(tmp_path: Path):
    cfg = SlurmConfig(account="project_TEST", logs_dir=str(tmp_path / "logs"),
                      results_dir=str(tmp_path / "results"))
    manifest = tmp_path / "manifest.jsonl"
    worker = tmp_path / "worker.sh"
    txt = render_sbatch(
        cfg=cfg, manifest_path=manifest, worker_path=worker,
        array_max=23, job_name="t",
    )
    assert "#SBATCH --account=project_TEST" in txt
    assert "#SBATCH --array=0-23%16" in txt
    assert str(worker) in txt
    assert "SLURM_ARRAY_TASK_ID" in txt


def test_worker_includes_container_and_venv(tmp_path: Path):
    cfg = SlurmConfig(
        account="project_TEST",
        container_sif="/scratch/sif/foo.sif",
        venv_activate="myvenv/bin/activate",
        binds=["/projappl/proj_X:/projappl/proj_X"],
    )
    manifest = tmp_path / "manifest.jsonl"
    txt = render_worker(cfg=cfg, manifest_path=manifest)
    assert "/scratch/sif/foo.sif" in txt
    assert "myvenv/bin/activate" in txt
    assert "--bind /projappl/proj_X:/projappl/proj_X" in txt
    assert "module load lumi-aif-singularity-bindings" in txt
    assert "python -m experiment.cli run-job" in txt
