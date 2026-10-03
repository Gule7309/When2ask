from when2ask.compatibility import (
    candidate_set_contains_gt,
    is_gt_compatible,
    make_oracle_partial_candidate,
)
from when2ask.models import Candidate, GroundTruthCall, UNK


def test_partial_candidate_can_represent_ground_truth():
    gt = GroundTruthCall("book_flight", {"from": "TPE", "to": "NRT", "class": "economy"})
    candidate = Candidate("book_flight", {"from": UNK, "to": "NRT", "class": "economy"})
    assert is_gt_compatible(candidate, gt)


def test_wrong_specified_argument_breaks_compatibility():
    gt = GroundTruthCall("book_flight", {"from": "TPE", "to": "NRT"})
    candidate = Candidate("book_flight", {"from": "KHH", "to": "NRT"})
    assert not is_gt_compatible(candidate, gt)


def test_wrong_tool_breaks_compatibility():
    gt = GroundTruthCall("cancel_booking", {"booking_id": "B1"})
    candidate = Candidate("retrieve_invoice", {"booking_id": UNK})
    assert not is_gt_compatible(candidate, gt)


def test_candidate_set_recall():
    gt = GroundTruthCall("touch", {"file_name": "a.txt"})
    candidates = [
        Candidate("mkdir", {"dir_name": UNK}),
        Candidate("touch", {"file_name": UNK}),
    ]
    assert candidate_set_contains_gt(candidates, gt)


def test_oracle_partial_candidate_does_not_reveal_unobserved_values():
    gt = GroundTruthCall("book_flight", {"from": "TPE", "to": "NRT", "class": "economy"})
    oracle = make_oracle_partial_candidate(gt, {"to": "NRT", "class": "economy"})
    assert oracle.arguments == {"from": UNK, "to": "NRT", "class": "economy"}
