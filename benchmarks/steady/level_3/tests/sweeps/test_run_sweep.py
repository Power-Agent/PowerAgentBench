# ABOUTME: Covers the sweep runner: campaign loading, queue construction, and resume behaviour.
# ABOUTME: The campaign file is the experiment definition, so its contract is pinned here too.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from restorebench.sweeps import run_sweep as runner


def _store(root: Path, cells: dict[str, int]) -> Path:
    cells_dir = root / "cells"
    cells_dir.mkdir(parents=True)
    for scenario_id, count in cells.items():
        for index in range(count):
            (cells_dir / f"{scenario_id}__cell{index}.json").write_text("{}", encoding="utf-8")
    (cells_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def test_published_campaigns_are_runnable_without_any_result_store():
    # The whole point of freezing the scenario list: a clean clone has no results, and the
    # campaign must still know exactly which cases it covers.
    campaigns = runner.load_campaigns()

    for name, campaign in campaigns.items():
        assert campaign.models, f"{name} names no models"
        assert campaign.configurations, f"{name} names no configurations"
        assert campaign.repetitions == 3
        assert campaign.max_runtime_seconds > 0
        assert campaign.data_dir.is_dir(), f"{name} points at a corpus that is not shipped"


def test_the_published_ieee118_campaigns_pin_the_same_46_scenarios():
    campaigns = runner.load_campaigns()
    anthropic = campaigns["ieee118-anthropic"]
    bedrock = campaigns["ieee118-bedrock"]

    # Both providers ran the identical subset; a comparison table is only meaningful if they did.
    assert anthropic.scenarios == bedrock.scenarios
    assert len(anthropic.scenarios) == 46
    assert anthropic.cells_per_scenario == 27


def test_frozen_scenarios_must_belong_to_the_held_out_split():
    campaign = runner.load_campaigns()["ieee118-anthropic"]

    # A campaign naming a memory-population scenario would leak the split; scope resolution
    # must refuse it rather than quietly run it.
    leaking = campaign.__class__(**{**campaign.__dict__, "scenarios": ("S0001",)})
    with pytest.raises(ValueError, match="outside the held-out split"):
        runner.campaign_scenarios(leaking)


def test_scenario_scope_falls_back_to_the_whole_held_out_split():
    campaign = runner.load_campaigns()["pegase89-bedrock"]

    assert campaign.scenarios is None
    assert len(runner.campaign_scenarios(campaign)) == 46


def test_pending_queue_drops_scenarios_already_complete(tmp_path):
    # Arrange
    store = _store(tmp_path / "bedrock", {"S0001": 9, "S0002": 3})

    # Act
    queue = runner.pending_queue(["S0001", "S0002", "S0003"], store, witness_lengths={}, cells_per_scenario=9)

    # Assert
    assert "S0001" not in queue
    assert set(queue) == {"S0002", "S0003"}


def test_pending_queue_runs_part_run_then_sequential_then_direct(tmp_path):
    # Arrange: S0003 is part-run, S0002 is sequential, S0001 and S0004 are direct.
    store = _store(tmp_path / "bedrock", {"S0003": 4})
    witness_lengths = {"S0001": 1, "S0002": 3, "S0003": 2, "S0004": 1}

    # Act
    queue = runner.pending_queue(
        ["S0001", "S0002", "S0003", "S0004"], store, witness_lengths=witness_lengths, cells_per_scenario=9
    )

    # Assert
    assert queue[0] == "S0003"
    assert queue[1] == "S0002"
    assert set(queue[2:]) == {"S0001", "S0004"}


def test_pending_queue_treats_an_empty_store_as_all_pending(tmp_path):
    store = tmp_path / "bedrock"

    assert runner.pending_queue(["S0001", "S0002"], store, witness_lengths={}, cells_per_scenario=9) == ["S0001", "S0002"]


def test_load_witness_lengths_reads_the_private_witness_file(tmp_path):
    # Arrange
    path = tmp_path / "witnesses.json"
    path.write_text(
        json.dumps(
            [
                {"scenario_id": "S0001", "maneuvers": [{"gen_id": 1}]},
                {"scenario_id": "S0002", "maneuvers": [{"gen_id": 1}, {"gen_id": 2}]},
            ]
        ),
        encoding="utf-8",
    )

    # Act
    lengths = runner.load_witness_lengths(path)

    # Assert
    assert lengths == {"S0001": 1, "S0002": 2}


def _cell(root: Path, scenario_id: str, name: str, status: str) -> Path:
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    path = cells / f"{scenario_id}__{name}.json"
    path.write_text(json.dumps({"status": status}), encoding="utf-8")
    return path


def test_failed_cell_paths_finds_only_provider_failures(tmp_path):
    # Arrange
    store = tmp_path / "bedrock"
    dead = _cell(store, "S0001", "config1__glm-5__rep0", "LLM_FAILURE")
    _cell(store, "S0001", "config2__glm-5__rep0", "SUCCESS")
    _cell(store, "S0001", "config3__glm-5__rep0", "BUDGET_EXHAUSTED")
    _cell(store, "S0001", "config1__kimi-k2.5__rep0", "TIMEOUT")

    # Act
    found = runner.failed_cell_paths(store)

    # Assert: TIMEOUT and BUDGET_EXHAUSTED are the model's results and must survive.
    assert found == [dead]


def test_failed_cell_paths_can_narrow_to_one_scenario(tmp_path):
    store = tmp_path / "bedrock"
    _cell(store, "S0001", "config1__glm-5__rep0", "LLM_FAILURE")
    wanted = _cell(store, "S0002", "config1__glm-5__rep0", "LLM_FAILURE")

    assert runner.failed_cell_paths(store, scenario_id="S0002") == [wanted]


def test_failed_cell_paths_ignores_unreadable_files(tmp_path):
    store = tmp_path / "bedrock"
    (store / "cells").mkdir(parents=True)
    (store / "cells" / "S0001__broken.json").write_text("{not json", encoding="utf-8")

    assert runner.failed_cell_paths(store) == []


def test_purge_failed_cells_deletes_them_and_leaves_the_rest(tmp_path):
    # Arrange
    store = tmp_path / "bedrock"
    dead = _cell(store, "S0001", "config1__glm-5__rep0", "LLM_FAILURE")
    kept = _cell(store, "S0001", "config2__glm-5__rep0", "SUCCESS")

    # Act
    purged = runner.purge_failed_cells(store)

    # Assert
    assert purged == [dead]
    assert not dead.exists()
    assert kept.exists()


def test_purge_failed_cells_is_a_no_op_on_a_healthy_store(tmp_path):
    store = tmp_path / "bedrock"
    _cell(store, "S0001", "config1__glm-5__rep0", "SUCCESS")

    assert runner.purge_failed_cells(store) == []


def _failed_cell(root: Path, name: str, detail: str) -> Path:
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    path = cells / f"S0001__{name}.json"
    path.write_text(
        json.dumps({"status": "LLM_FAILURE", "failure_feedback": [{"detail": detail}]}),
        encoding="utf-8",
    )
    return path


def test_fatal_provider_reason_spots_an_exhausted_credit(tmp_path):
    # Arrange
    dead = _failed_cell(tmp_path, "a", "LLM call failed: Your credit balance is too low")

    # Act
    reason = runner.fatal_provider_reason([dead])

    # Assert
    assert reason is not None
    assert "credit balance" in reason.lower()


def test_fatal_provider_reason_spots_a_dead_credential(tmp_path):
    dead = _failed_cell(tmp_path, "a", "An error occurred (UnrecognizedClientException): token expired")

    assert runner.fatal_provider_reason([dead]) is not None


def test_fatal_provider_reason_ignores_a_passing_outage(tmp_path):
    # A ServiceUnavailable is exactly what the retry loop exists to wait out.
    dead = _failed_cell(tmp_path, "a", "An error occurred (ServiceUnavailableException): try later")

    assert runner.fatal_provider_reason([dead]) is None


def test_fatal_provider_reason_ignores_files_already_gone(tmp_path):
    assert runner.fatal_provider_reason([tmp_path / "cells" / "absent.json"]) is None


def test_select_batch_returns_the_whole_queue_by_default():
    queue = ["S0001", "S0002", "S0003"]

    assert runner.select_batch(queue) == queue


def test_select_batch_takes_the_first_n_scenarios():
    queue = ["S0001", "S0002", "S0003"]

    assert runner.select_batch(queue, first=2) == ["S0001", "S0002"]


def test_select_batch_tolerates_a_batch_larger_than_the_queue():
    assert runner.select_batch(["S0001"], first=10) == ["S0001"]


def test_select_batch_narrows_to_a_single_scenario():
    queue = ["S0001", "S0002", "S0003"]

    assert runner.select_batch(queue, only="S0002") == ["S0002"]


def test_select_batch_rejects_a_scenario_that_is_not_pending():
    with pytest.raises(ValueError, match="S0009"):
        runner.select_batch(["S0001"], only="S0009")


def test_select_batch_applies_only_before_first():
    # --only names a scenario; truncating first would silently run a different one.
    queue = ["S0001", "S0002", "S0003"]

    assert runner.select_batch(queue, only="S0003", first=1) == ["S0003"]


def test_each_campaign_writes_into_its_own_store():
    # Two campaigns sharing a results directory would interleave cells from different model
    # suites under one manifest, and the aggregate would silently mix them.
    campaigns = runner.load_campaigns()
    stores = [campaign.results_dir for campaign in campaigns.values()]

    assert len(set(stores)) == len(stores)


def test_campaign_models_resolve_to_a_known_provider():
    from restorebench.llm import models

    for name, campaign in runner.load_campaigns().items():
        providers = {models.provider_for(model_id) for model_id in campaign.models}
        # A suite spanning two transports in one campaign would make a single credential
        # insufficient to run it, which is exactly what the campaign split avoids.
        assert len(providers) == 1, f"{name} mixes providers: {providers}"
        for model_id in campaign.models:
            assert models.model_slug(model_id), f"{name} names an unslugged model {model_id}"


def test_campaign_runtime_limit_is_installed_before_a_sweep(monkeypatch):
    # MAX_RUNTIME_SECONDS is wall clock and a TIMEOUT is recorded as the model's own result, so
    # re-running a published campaign under a different limit measures a different thing.
    from restorebench.eval import harness
    from restorebench.eval.store import RunKey

    monkeypatch.setattr(harness, "_config_for_key", runner._original_config_for_key)
    runner.install_runtime_limit(1234)
    config = harness._config_for_key(RunKey("S0001", "claude-opus-5", 2, 0))

    assert config.MAX_RUNTIME_SECONDS == 1234
