"""Unit tests for UI schema synchronization, parsing, importing and exporting in app.py."""

import json
from pathlib import Path

import pytest

from app import (
    MAX_FIELDS,
    apply_items_to_form_updates,
    parse_raw_json_schema,
    serialize_form_to_json,
)


def test_parse_raw_json_schema_valid_list():
    raw = json.dumps(
        [
            {"key": "invoice_id", "description": "Invoice number", "type": "id"},
            {"key": "total", "description": "Total amount", "type": "money"},
        ]
    )
    items = parse_raw_json_schema(raw)
    assert len(items) == 2
    assert items[0] == {"key": "invoice_id", "description": "Invoice number", "type": "id"}
    assert items[1] == {"key": "total", "description": "Total amount", "type": "money"}


def test_parse_raw_json_schema_single_dict():
    raw = json.dumps(
        {"key": "vendor", "description": "Vendor organization name", "type": "organization"}
    )
    items = parse_raw_json_schema(raw)
    assert len(items) == 1
    assert items[0]["key"] == "vendor"
    assert items[0]["type"] == "organization"


def test_parse_raw_json_schema_alternative_fields():
    raw = json.dumps(
        [
            {"field": "due_date", "prompt": "When is it due?", "type": "date"},
            {"name": "buyer", "question": "Who is the buyer?"},
        ]
    )
    items = parse_raw_json_schema(raw)
    assert len(items) == 2
    assert items[0]["key"] == "due_date"
    assert items[0]["description"] == "When is it due?"
    assert items[0]["type"] == "date"
    assert items[1]["key"] == "buyer"
    assert items[1]["description"] == "Who is the buyer?"
    assert items[1]["type"] == "string"


def test_parse_raw_json_schema_type_mapping():
    raw = json.dumps(
        [
            {"key": "val1", "description": "test", "type": "para"},
            {"key": "val2", "description": "test", "type": "tarih"},
            {"key": "val3", "description": "test", "type": "unknown_type"},
        ]
    )
    items = parse_raw_json_schema(raw)
    assert items[0]["type"] == "money"
    assert items[1]["type"] == "date"
    assert items[2]["type"] == "string"


def test_parse_raw_json_schema_empty():
    assert parse_raw_json_schema("") == []
    assert parse_raw_json_schema("   ") == []


def test_parse_raw_json_schema_invalid():
    with pytest.raises(ValueError, match="Invalid JSON syntax"):
        parse_raw_json_schema("not a json")

    with pytest.raises(ValueError, match="Schema must be a JSON array"):
        parse_raw_json_schema('"just a string"')


def test_serialize_form_to_json():
    q_vals = ["What is the total?", "Invoice date", ""]
    k_vals = ["total_amount", "inv_date", ""]
    t_vals = ["money", "date", "string"]
    vis_vals = [True, True, False]

    json_str = serialize_form_to_json(q_vals, k_vals, t_vals, vis_vals)
    data = json.loads(json_str)
    assert len(data) == 2
    assert data[0]["key"] == "total_amount"
    assert data[0]["type"] == "money"
    assert data[1]["key"] == "inv_date"
    assert data[1]["type"] == "date"


def test_serialize_form_to_json_empty():
    json_str = serialize_form_to_json([], [], [])
    data = json.loads(json_str)
    assert len(data) == 1
    assert data[0]["key"] == "example_field"


def test_apply_items_to_form_updates():
    items = [
        {"key": "k1", "description": "desc1", "type": "money"},
        {"key": "k2", "description": "desc2", "type": "date"},
    ]
    new_vis, grp_ups, q_ups, k_ups, t_ups, json_str = apply_items_to_form_updates(items)

    assert len(new_vis) == MAX_FIELDS
    assert new_vis[0] is True
    assert new_vis[1] is True
    assert new_vis[2] is False

    assert len(grp_ups) == MAX_FIELDS
    assert len(q_ups) == MAX_FIELDS
    assert len(k_ups) == MAX_FIELDS
    assert len(t_ups) == MAX_FIELDS

    data = json.loads(json_str)
    assert len(data) == 2
    assert data[0]["key"] == "k1"


def test_export_file_generation(tmp_path: Path):
    items = [{"key": "test_key", "description": "test description", "type": "string"}]
    export_path = tmp_path / "docneedle_schema_test.json"
    export_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    assert export_path.exists()
    loaded = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert loaded[0]["key"] == "test_key"


def test_import_file_parsing(tmp_path: Path):
    sample = [
        {"key": "inv_no", "description": "Invoice number", "type": "id"},
        {"key": "total", "description": "Total with tax", "type": "money"},
    ]
    file_path = tmp_path / "schema.json"
    file_path.write_text(json.dumps(sample), encoding="utf-8")

    content = file_path.read_text(encoding="utf-8")
    parsed = parse_raw_json_schema(content)
    assert len(parsed) == 2
    new_vis, grp_ups, q_ups, k_ups, t_ups, json_str = apply_items_to_form_updates(parsed)
    assert new_vis[0] is True
    assert new_vis[1] is True
    assert new_vis[2] is False
    assert json.loads(json_str)[0]["key"] == "inv_no"


def test_perform_extraction_ui_measures_end_to_end_time():
    import asyncio
    from unittest.mock import patch

    from app import perform_extraction_ui

    async def mock_async_extract(*args, **kwargs):
        await asyncio.sleep(0.05)
        return (
            "<div>Mock HTML</div>",
            {"total_items": 1, "found_items": 1, "processing_time_ms": 1.0, "results": []},
            [],
            "Status",
        )

    with patch("app._async_perform_extraction_ui", side_effect=mock_async_extract):
        html_out, resp_dict, _, status = perform_extraction_ui(
            None, [], "form", "", [True], "test_q", *([""] * 29)
        )
        assert resp_dict.get("processing_time_ms") >= 45.0
        assert "Processing Time" in html_out
        assert "Extraction complete: 1 of 1 items found" in status
