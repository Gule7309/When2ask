from when2ask.provider import OpenAIJsonGenerator


def test_parse_json_accepts_plain_object():
    payload = OpenAIJsonGenerator._parse_json('{"candidates": []}')
    assert payload == {"candidates": []}


def test_parse_json_accepts_fenced_object():
    text = '```json\n{"candidates": []}\n```'
    payload = OpenAIJsonGenerator._parse_json(text)
    assert payload == {"candidates": []}
