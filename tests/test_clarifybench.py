import json

from when2ask.clarifybench import load_sample


def test_loads_current_clarifybench_schema(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {
                "user_query": "Create a file.",
                "ground_truth_tool_calls": [
                    {"tool_name": "touch", "parameters": {"file_name": "a.txt"}}
                ],
                "primary_api": "GorillaFileSystem",
            }
        ),
        encoding="utf-8",
    )
    sample = load_sample(path)
    assert sample.sample_id == "sample"
    assert sample.ground_truth_tool_calls[0].tool_name == "touch"
