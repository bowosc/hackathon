"""Bundle snapshots stay fixed; checkout keeps each bundle's generation instructions."""
import json

import pytest
from fastapi.testclient import TestClient

from backend import bundles, catalog, datasets
from backend.main import app

client = TestClient(app)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bundles, "ROOT", tmp_path)
    monkeypatch.setattr(datasets, "ROOT", tmp_path)
    monkeypatch.setitem(bundles.SETTINGS, "bundles_directory", "bundles")
    monkeypatch.setitem(datasets.SETTINGS, "requests_directory", "requests")
    return tmp_path


def draft(**plan):
    return {"prompt": "Hand interactions for a banana plantation", "selection": {
        "plan": {"summary": "Hand interactions", "modifications": ["object_swap"], "target_object": "bananas", **plan}}}


def save(body):
    response = client.post("/api/bundles", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_snapshot_includes_every_page_and_keeps_exclusions_and_settings(storage, monkeypatch):
    clips = catalog.catalog()
    body = draft(background="a banana plantation", modifications=["object_swap", "background"], variants_per_clip=5)
    body["selection"]["excluded_ids"] = [clips[0]["id"]]
    preview = client.post("/api/dataset-preview?limit=1", json=body["selection"]).json()
    saved = save(body)
    assert saved["clip_count"] == preview["selected_count"] == len(clips) - 1 > len(preview["clips"])
    snapshot = json.loads((storage / "bundles" / f"{saved['id']}.json").read_text())
    assert set(snapshot["source_clip_ids"]) == {clip["id"] for clip in clips[1:]}
    assert snapshot["selection"] == datasets.Selection.model_validate(body["selection"]).model_dump()
    assert "source_clip_ids" not in saved   # full selections stay on the server, not in browser storage

    # New matching footage does not silently expand a saved bundle.
    monkeypatch.setattr(bundles, "catalog", lambda: [*clips, {**clips[0], "id": "new-matching-clip"}])
    result = client.post("/api/bundle-checkout-preview", json={"bundle_ids": [saved["id"]]}).json()
    assert result["totals"]["clip_count"] == len(clips) - 1
    assert result["bundles"][0]["plan"]["variants_per_clip"] == 5


def test_several_bundles_make_one_checkout_with_separate_generation_specs(storage):
    first = save(draft())
    second = save(draft(modifications=["lighting", "upscale"], lighting="low_light", variants_per_clip=2))
    ids = [first["id"], second["id"]]
    preview = client.post("/api/bundle-checkout-preview", json={"bundle_ids": ids}).json()
    assert preview["totals"]["bundle_count"] == 2
    assert preview["totals"]["clip_count"] == first["clip_count"] + second["clip_count"]
    assert preview["totals"]["unique_clip_count"] == first["clip_count"]
    assert preview["totals"]["duration_seconds"] == pytest.approx(first["duration_seconds"] + second["duration_seconds"])
    response = client.post("/api/dataset-checkouts", json={"bundle_ids": ids, "lab": "Robotics lab", "email": "buyer@example.test"})
    assert response.status_code == 201, response.text
    manifest = response.json()
    assert manifest["totals"] == preview["totals"]
    assert manifest["payment_status"] == "not_requested" and manifest["generation_status"] == "not_started"
    assert [item["bundle_id"] for item in manifest["bundles"]] == ids
    assert manifest["bundles"][0]["generation_spec"]["target_object"] == "bananas"
    assert manifest["bundles"][1]["generation_spec"]["lighting"] == "low_light"
    assert all(item["request"]["email"] == "buyer@example.test" for item in manifest["bundles"])
    files = list((storage / "requests").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text()) == manifest


def test_checkout_does_not_save_a_partial_order_when_a_source_disappears(storage, monkeypatch):
    saved = save(draft())
    monkeypatch.setattr(bundles, "catalog", lambda: catalog.catalog()[1:])
    response = client.post("/api/dataset-checkouts", json={"bundle_ids": [saved["id"]], "email": "buyer@example.test"})
    assert response.status_code == 409
    assert not (storage / "requests").exists()


def test_bad_or_duplicate_bundle_references_are_rejected(storage):
    saved = save(draft())
    for ids, code in [([], 422), (["../../config/settings"], 422), ([saved["id"], saved["id"]], 422),
                      (["bundle-" + "0" * 32], 409), ([saved["id"]] * (catalog.SETTINGS["max_checkout_bundles"] + 1), 422)]:
        assert client.post("/api/bundle-checkout-preview", json={"bundle_ids": ids}).status_code == code
    assert client.post("/api/dataset-checkouts", json={"bundle_ids": [saved["id"]], "email": "invalid"}).status_code == 422


def test_empty_or_incomplete_bundles_are_not_saved(storage):
    bodies = [draft(target_object=""), draft(keywords="teleporting forklift"),
              {"prompt": "  ", "selection": {"plan": {"summary": "Empty prompt"}}},
              {"prompt": "No plan", "selection": {}}]
    for body in bodies:
        assert client.post("/api/bundles", json=body).status_code == 422
    assert not (storage / "bundles").exists()


def test_bundle_storage_is_bounded_and_cannot_leave_the_repo(storage, monkeypatch):
    monkeypatch.setitem(bundles.SETTINGS, "max_saved_bundles", 1)
    save(draft())
    assert client.post("/api/bundles", json=draft()).status_code == 503
    monkeypatch.setitem(bundles.SETTINGS, "bundles_directory", "../outside")
    assert client.post("/api/bundles", json=draft()).status_code == 500
