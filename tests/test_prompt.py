from barthel import prompt as P


def test_plain_json_with_string_scores():
    out = P.parse_output('{"essen": "10", "sich_waschen": "not_assessed"}')
    assert out == {"essen": 10, "sich_waschen": "not_assessed"}


def test_markdown_fence_and_surrounding_text_are_accepted():
    fenced = '```json\n{"essen": "5"}\n```'
    chatty = 'Hier ist das Ergebnis: {"essen": "5"} Viel Erfolg.'
    assert P.parse_output(fenced) == {"essen": 5}
    assert P.parse_output(chatty) == {"essen": 5}


def test_content_is_not_repaired():
    assert P.parse_output('{"essen": "5",}') is None  # trailing comma
    assert P.parse_output("kein JSON") is None
    assert P.parse_output("") is None
    assert P.parse_output(None) is None
    assert P.parse_output("[1, 2]") is None  # not an object


def test_invalid_values_pass_through_for_eval_to_flag():
    out = P.parse_output('{"essen": "7", "sich_waschen": "yes"}')
    assert out == {"essen": 7, "sich_waschen": "yes"}  # evaluation.py counts these as parse failures


def test_schema_and_splits():
    assert len(P.load_split("train")) == 180 and len(P.load_split("val")) == 20 and len(P.load_split("test")) == 40
    ids = {n["note_id"] for n in P.load_split("train")} & {n["note_id"] for n in P.load_split("val")}
    assert not ids  # no leakage between train and val
    schema = P.output_schema()
    assert set(schema["required"]) == set(schema["properties"])
