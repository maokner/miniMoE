import json

import pytest

from eval_utils import append_jsonl, completed_indices, merge_rank_parts, part_path


def test_distributed_merge_is_sorted_and_deduplicated(tmp_path):
    rank_one = tmp_path / "rank1.jsonl"
    rank_zero = tmp_path / "rank0.jsonl"
    append_jsonl(rank_one, [{"index": 3, "value": "d"}, {"index": 1, "value": "b"}])
    append_jsonl(rank_zero, [{"index": 2, "value": "c"}, {"index": 0, "value": "a"}])
    merged = merge_rank_parts([rank_one, rank_zero])
    assert [record["index"] for record in merged] == [0, 1, 2, 3]

    append_jsonl(rank_zero, [{"index": 0, "value": "a"}])
    assert len(merge_rank_parts([rank_zero, rank_one])) == 4


def test_resume_indices_do_not_create_duplicates(tmp_path):
    part = tmp_path / "rank.jsonl"
    append_jsonl(part, [{"index": 0}, {"index": 2}])
    pending = [index for index in range(5) if index not in completed_indices(part)]
    append_jsonl(part, ({"index": index} for index in pending))
    records = merge_rank_parts([part])
    assert [record["index"] for record in records] == list(range(5))
    assert len(part.read_text().splitlines()) == 5


def test_torn_final_line_is_dropped_and_file_repaired(tmp_path):
    part = tmp_path / "rank.jsonl"
    append_jsonl(part, [{"index": 0}])
    with part.open("a") as handle:
        handle.write('{"index": 1')  # killed mid-append: no trailing newline
    assert completed_indices(part) == {0}
    append_jsonl(part, [{"index": 1}])
    assert [record["index"] for record in merge_rank_parts([part])] == [0, 1]


def test_part_paths_differ_across_world_sizes(tmp_path):
    output = tmp_path / "result.json"
    assert part_path(output, 0, 1) != part_path(output, 0, 2)
    assert part_path(output, 0, 2) == part_path(output, 0, 2)


def test_conflicting_duplicate_is_rejected(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"index": 1, "value": "a"}) + "\n")
    second.write_text(json.dumps({"index": 1, "value": "b"}) + "\n")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        merge_rank_parts([first, second])
