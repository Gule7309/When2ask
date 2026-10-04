from when2ask.probe_analysis import summarize_initial_probe


def _record(sample_id, gt_present, best_gt, viability, action="execute"):
    tool = "cancel_booking" if best_gt else "retrieve_invoice"
    args = {"booking_id": "<UNK>"}
    return {
        "sample_id": sample_id,
        "gt_present": gt_present,
        "candidates": [
            {
                "tool_name": tool,
                "arguments": args,
                "viability": viability,
                "gt_compatible": best_gt,
            }
        ],
        "pre_question_decision": {
            "action": action,
            "best_tool": tool,
            "best_arguments": args,
            "best_viability": viability,
        },
    }


def test_summary_separates_candidate_recall_from_wrong_best_confidence():
    records = [
        _record("present", True, True, 1.0),
        _record("absent", False, False, 0.9),
    ]
    report = summarize_initial_probe(records, [0.8, 0.95])

    assert report["candidate_recall"] == 0.5
    assert report["n_gt_absent"] == 1
    assert report["mean_best_viability_gt_absent"] == 0.9
    assert report["high_viability_wrong_best_rate_all"]["0.8"] == 0.5
    assert (
        report["high_viability_wrong_best_rate_given_gt_absent"]["0.8"]
        == 1.0
    )
    assert (
        report["high_viability_wrong_best_rate_given_gt_absent"]["0.95"]
        == 0.0
    )
