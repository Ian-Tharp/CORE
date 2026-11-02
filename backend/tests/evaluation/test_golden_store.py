import pytest

from app.evaluation import DuplicateCaseError, FaultReport, GoldenSet
from app.evaluation.models import Contains, ExactMatch, GoldenCase, MaxLatency
from app.evaluation.signature import case_signature


def case(id_="a", text="hello", ev=None):
    return GoldenCase(id=id_, input=text, evaluators=ev or [Contains(value="x")])


def test_add_and_load_roundtrip(tmp_path):
    gs = GoldenSet(tmp_path / "sub" / "g.jsonl")
    gs.add(case("a", "one"))
    gs.add(case("b", "two"))
    res = gs.load()
    assert [c.id for c in res.cases] == ["a", "b"] and res.skipped == []


def test_missing_file_loads_empty(tmp_path):
    assert GoldenSet(tmp_path / "nope.jsonl").load().cases == []


def test_duplicate_signature_rejected_despite_whitespace_and_case(tmp_path):
    gs = GoldenSet(tmp_path / "g.jsonl")
    gs.add(case("a", "Book  a ROOM"))
    with pytest.raises(DuplicateCaseError):
        gs.add(case("b", " book a room "))
    assert len(gs.load().cases) == 1


def test_duplicate_id_rejected_same_input_different_checks_allowed(tmp_path):
    gs = GoldenSet(tmp_path / "g.jsonl")
    gs.add(case("a", "q"))
    with pytest.raises(DuplicateCaseError):
        gs.add(case("a", "different"))
    gs.add(case("c", "q", [MaxLatency(ms=5)]))  # same prompt, other check
    assert len(gs.load().cases) == 2


def test_signature_ignores_evaluator_order():
    e1, e2 = Contains(value="a"), MaxLatency(ms=5)
    assert case_signature("q", [e1, e2]) == case_signature("q", [e2, e1])


def test_corrupt_lines_skipped_rest_kept(tmp_path):
    p = tmp_path / "g.jsonl"
    good = case("a", "one").model_dump_json()
    good2 = case("b", "two").model_dump_json()
    p.write_text(
        good
        + "\n{broken\n"
        + '{"id":"z","input":"q","evaluators":[]}\n\n'
        + good2
        + "\n"
    )
    res = GoldenSet(p).load()
    assert [c.id for c in res.cases] == ["a", "b"]
    assert [s.line_no for s in res.skipped] == [2, 3]
    assert "evaluators" in res.skipped[1].reason


def test_duplicates_in_file_collapse_on_load(tmp_path):
    p = tmp_path / "g.jsonl"
    line = case("a", "one").model_dump_json()
    p.write_text(line + "\n" + line.replace('"a"', '"b"', 1) + "\n")
    res = GoldenSet(p).load()
    assert len(res.cases) == 1 and "duplicate of line 1" in res.skipped[0].reason


def test_append_repairs_truncated_last_line(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(case("a", "one").model_dump_json())  # no trailing newline
    GoldenSet(p).add(case("b", "two"))
    assert [c.id for c in GoldenSet(p).load().cases] == ["a", "b"]


def test_append_fault_defaults_to_exact_match_and_dedupes(tmp_path):
    gs = GoldenSet(tmp_path / "g.jsonl")
    fault = FaultReport(input="2+2?", expected="4", actual="5")
    c = gs.append_fault(fault, tags=["math"])
    assert c.source == "fault" and c.fault_id == fault.fault_id
    assert c.evaluators == [ExactMatch(value="4")] and c.tags == ["math"]
    with pytest.raises(DuplicateCaseError):
        gs.append_fault(FaultReport(input="2+2?  ", expected="4"))


def test_append_fault_needs_expected_or_evaluators(tmp_path):
    gs = GoldenSet(tmp_path / "g.jsonl")
    with pytest.raises(ValueError):
        gs.append_fault(FaultReport(input="x"))
    c = gs.append_fault(FaultReport(input="x"), evaluators=[Contains(value="y")])
    assert c.evaluators == [Contains(value="y")]


def test_case_requires_evaluator_and_input():
    with pytest.raises(ValueError):
        GoldenCase(id="a", input="x", evaluators=[])
    with pytest.raises(ValueError):
        GoldenCase(id="a", input="", evaluators=[Contains(value="x")])
