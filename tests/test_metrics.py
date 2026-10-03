from when2ask.metrics import (
    candidate_recall,
    conditional_execution_accuracy,
    high_confidence_wrong_execution_rate,
)
from when2ask.models import Candidate, DecisionRecord, GroundTruthCall, UNK


def _record(sample_id, candidates, gt, executed, correct):
    return DecisionRecord(
        sample_id=sample_id,
        candidates=tuple(candidates),
        ground_truth=gt,
        executed_candidate=executed,
        executed_correctly=correct,
    )


def test_candidate_recall_and_conditional_accuracy():
    gt = GroundTruthCall("touch", {"file_name": "a.txt"})
    good = Candidate("touch", {"file_name": UNK}, confidence=0.9)
    bad = Candidate("mkdir", {"dir_name": UNK}, confidence=0.9)
    records = [
        _record("a", [good], gt, good, True),
        _record("b", [bad], gt, bad, False),
    ]
    assert candidate_recall(records) == 0.5
    assert conditional_execution_accuracy(records, gt_present=True) == 1.0
    assert conditional_execution_accuracy(records, gt_present=False) == 0.0


def test_high_confidence_wrong_execution_rate_uses_all_records_as_denominator():
    gt = GroundTruthCall("touch", {"file_name": "a.txt"})
    wrong_high = Candidate("mkdir", {"dir_name": UNK}, confidence=0.95)
    wrong_low = Candidate("mkdir", {"dir_name": UNK}, confidence=0.60)
    records = [
        _record("a", [wrong_high], gt, wrong_high, False),
        _record("b", [wrong_low], gt, wrong_low, False),
    ]
    assert high_confidence_wrong_execution_rate(records, threshold=0.8) == 0.5
