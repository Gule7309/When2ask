import pytest

from when2ask.interventions import forced_miss, oracle_complete
from when2ask.models import Candidate, GroundTruthCall, UNK


def test_oracle_complete_adds_compatible_candidate_when_missing():
    gt = GroundTruthCall("cancel_booking", {"booking_id": "B1"})
    natural = [Candidate("retrieve_invoice", {"booking_id": UNK})]
    oracle = Candidate("cancel_booking", {"booking_id": UNK})
    result = oracle_complete(natural, gt, oracle)
    assert result.changed
    assert len(result.candidates) == 2


def test_oracle_complete_is_noop_when_gt_already_present():
    gt = GroundTruthCall("cancel_booking", {"booking_id": "B1"})
    natural = [Candidate("cancel_booking", {"booking_id": UNK})]
    result = oracle_complete(
        natural, gt, Candidate("cancel_booking", {"booking_id": UNK})
    )
    assert not result.changed
    assert tuple(natural) == result.candidates


def test_forced_miss_removes_ground_truth_and_can_add_decoy():
    gt = GroundTruthCall("cancel_booking", {"booking_id": "B1"})
    natural = [
        Candidate("cancel_booking", {"booking_id": UNK}),
        Candidate("retrieve_invoice", {"booking_id": UNK}),
    ]
    decoy = Candidate("contact_customer_support", {"booking_id": UNK})
    result = forced_miss(natural, gt, replacement=decoy)
    assert result.changed
    assert all(c.tool_name != "cancel_booking" for c in result.candidates)
    assert len(result.candidates) == 2


def test_forced_miss_rejects_gt_compatible_replacement():
    gt = GroundTruthCall("cancel_booking", {"booking_id": "B1"})
    with pytest.raises(ValueError):
        forced_miss(
            [Candidate("cancel_booking", {"booking_id": UNK})],
            gt,
            replacement=Candidate("cancel_booking", {"booking_id": UNK}),
        )
