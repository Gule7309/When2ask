import json

from when2ask.trace_io import load_jsonl


def test_load_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "ground_truth": {
                    "tool_name": "touch",
                    "parameters": {"file_name": "a.txt"},
                },
                "candidates": [
                    {
                        "tool_name": "touch",
                        "arguments": {"file_name": "<UNK>"},
                        "confidence": 0.8,
                    }
                ],
                "executed_candidate": {
                    "tool_name": "touch",
                    "arguments": {"file_name": "a.txt"},
                    "confidence": 0.9,
                },
                "executed_correctly": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert len(records) == 1
    assert records[0].sample_id == "s1"
    assert records[0].candidates[0].confidence == 0.8
