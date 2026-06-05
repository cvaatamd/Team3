from agents.contracts import ExperimentPlan


def test_plan_enumeration_count():
    plan = ExperimentPlan(
        target_endpoints=["HLM CLint", "MBPB"],
        arms=["baseline", "intra_task"],
        n_grid=[25, 50, None],
        seeds=[0, 1, 2],
        mechanism_by_arm={"baseline": "gbm_baseline", "intra_task": "mt_cotrain"},
    )
    jobs = plan.enumerate_jobs()
    assert len(jobs) == 2 * 2 * 3 * 3


def test_plan_job_ids_unique_and_stable():
    plan = ExperimentPlan(
        target_endpoints=["HLM CLint"],
        arms=["baseline"],
        n_grid=[25],
        seeds=[0],
        mechanism_by_arm={"baseline": "gbm_baseline"},
    )
    jobs = plan.enumerate_jobs()
    assert len(jobs) == 1
    assert jobs[0].job_id == "HLM_CLint__baseline__25__s0"
