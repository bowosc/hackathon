"""Payment verification and entitlement checks use mocked Stripe calls, never real charges."""
import hashlib
import hmac
import io
import json
import time
from zipfile import ZipFile

import pytest
import stripe
from fastapi.testclient import TestClient

from backend import bundles, catalog, commerce, datasets
from backend.main import app

client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


@pytest.fixture
def store(tmp_path, monkeypatch):
    for module in [commerce, bundles, datasets]:
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(commerce, "CONFIG", {**commerce.CONFIG, "checkout_mode": "purchase", "payment_provider": "demo",
                                          "public_url": "http://127.0.0.1", "orders_directory": "purchases"})
    return tmp_path


def draft():
    return {"prompt": "Hand interactions", "selection": {"plan": {"summary": "Hands", "modifications": []}}}


def buy(body=None):
    body = body or {"dataset": draft()}
    preview = client.post("/api/purchase-preview", json=body)
    assert preview.status_code == 200, preview.text
    result = client.post("/api/purchases", json={**body, "email": "buyer@example.test", "lab": "Test lab",
        "accept_available_versions": True, "expected_amount_cents": preview.json()["amount_cents"]})
    assert result.status_code == 201, result.text
    return result.json()


def auth(purchase):
    return {"Authorization": f"Bearer {purchase['access_token']}"}


def test_demo_purchase_delivers_real_video_and_annotations_only_after_checkout(store):
    purchase = buy()
    endpoint = f"/api/purchases/{purchase['id']}"
    assert client.get(endpoint + "/download", headers=auth(purchase)).status_code == 402
    assert client.get(endpoint, headers={"Authorization": "Bearer incorrect"}).status_code == 404
    assert client.post(endpoint + "/demo-pay", headers=auth(purchase)).status_code == 200
    status = client.get(endpoint, headers=auth(purchase))
    assert status.json()["ready"] and status.json()["status"] == "demo_paid"
    assert "HttpOnly" in status.headers["set-cookie"] and "SameSite=strict" in status.headers["set-cookie"]
    download = client.get(endpoint + "/download")   # scoped cookie permits a normal browser download
    assert download.status_code == 200 and download.headers["content-type"] == "application/zip"
    with ZipFile(io.BytesIO(download.content)) as archive:
        manifest = json.loads(archive.read("dataset.json"))
        assert len(manifest["clips"]) == len(catalog.catalog())
        assert len([name for name in archive.namelist() if name.endswith(".mp4")]) == purchase["quote"]["video_count"]
        assert any(name.endswith("labels.json") for name in archive.namelist())
        assert archive.read(manifest["clips"][0]["video_url"])[4:8] == b"ftyp"
        assert manifest["requested_transformations_status"] == "not_generated"
        assert all(clip["attribution"] and clip["license"] for clip in manifest["clips"])
    commerce.update_order(purchase["id"], expires_at=time.time() - 1)
    assert client.get(endpoint + "/download", headers=auth(purchase)).status_code == 410


def test_bundles_are_delivered_separately_but_overlapping_source_files_are_billed_once(store):
    first = client.post("/api/bundles", json=draft()).json()
    changed = draft()
    changed["selection"]["plan"].update(modifications=["object_swap"], target_object="bananas")
    second = client.post("/api/bundles", json=changed).json()
    body = {"bundle_ids": [first["id"], second["id"]]}
    quote = client.post("/api/purchase-preview", json=body).json()
    single_quote = client.post("/api/purchase-preview", json={"dataset": draft()}).json()
    assert quote["amount_cents"] == single_quote["amount_cents"]
    assert quote["clip_count"] == quote["unique_clip_count"] * 2
    assert quote["requires_custom_ack"]
    assert client.post("/api/purchases", json={**body, "email": "buyer@example.test", "expected_amount_cents": quote["amount_cents"]}).status_code == 422
    purchase = buy(body)
    _, order = commerce.read_order(purchase["id"])
    assert len(order["manifest"]["bundles"]) == 2
    assert order["manifest"]["bundles"][1]["generation_spec"]["target_object"] == "bananas"


@pytest.fixture
def stripe_api(store, monkeypatch):
    monkeypatch.setitem(commerce.CONFIG, "payment_provider", "stripe")
    monkeypatch.setattr(commerce, "secret", lambda name: {"STRIPE_SECRET_KEY": "sk_test_mock", "STRIPE_WEBHOOK_SECRET": "whsec_test"}.get(name))
    state = {}
    def create(**kwargs):
        state.update(kwargs)
        state["session"] = stripe.StripeObject.construct_from({"id": "cs_test_mock", "url": "https://checkout.stripe.com/test",
            "livemode": False, "metadata": kwargs["metadata"], "client_reference_id": kwargs["client_reference_id"],
            "amount_total": kwargs["line_items"][0]["price_data"]["unit_amount"], "currency": "usd",
            "payment_status": "unpaid", "status": "open"}, "sk_test_mock")
        return state["session"]
    monkeypatch.setattr(stripe.checkout.Session, "create", create)
    monkeypatch.setattr(stripe.checkout.Session, "retrieve", lambda *args, **kwargs: state["session"])
    return state


def test_stripe_payment_is_verified_server_side_and_fulfilled_once(stripe_api):
    purchase = buy()
    endpoint = f"/api/purchases/{purchase['id']}"
    assert stripe_api["line_items"][0]["price_data"]["unit_amount"] == purchase["quote"]["amount_cents"]
    assert stripe_api["idempotency_key"] == purchase["id"]
    assert client.get(endpoint, headers=auth(purchase)).json()["ready"] is False
    assert client.get(endpoint + "/download", headers=auth(purchase)).status_code == 402
    assert client.post(endpoint + "/demo-pay", headers=auth(purchase)).status_code == 403
    stripe_api["session"].update({"payment_status": "paid", "status": "complete"})
    # A mismatching payment amount never grants access.
    correct = stripe_api["session"]["amount_total"]
    stripe_api["session"]["amount_total"] = 1
    assert client.get(endpoint, headers=auth(purchase)).status_code == 409
    stripe_api["session"]["amount_total"] = correct
    assert client.get(endpoint, headers=auth(purchase)).json()["ready"]
    _, first = commerce.read_order(purchase["id"])
    commerce.fulfill("cs_test_mock")
    _, repeated = commerce.read_order(purchase["id"])
    assert repeated["paid_at"] == first["paid_at"] and repeated["expires_at"] == first["expires_at"]
    assert client.get(endpoint + "/download", headers=auth(purchase)).status_code == 200


def test_signed_webhook_unlocks_download_without_a_browser_return(stripe_api):
    purchase = buy()
    stripe_api["session"].update({"payment_status": "paid", "status": "complete"})
    payload = json.dumps({"id": "evt_test", "object": "event", "type": "checkout.session.completed",
                          "data": {"object": {"id": "cs_test_mock"}}}).encode()
    stamp = str(int(time.time()))
    signature = hmac.new(b"whsec_test", stamp.encode() + b"." + payload, hashlib.sha256).hexdigest()
    assert client.post("/api/stripe/webhook", content=payload, headers={"Stripe-Signature": f"t={stamp},v1=invalid"}).status_code == 400
    headers = {"Stripe-Signature": f"t={stamp},v1={signature}"}
    assert client.post("/api/stripe/webhook", content=payload, headers=headers).status_code == 200
    assert client.post("/api/stripe/webhook", content=payload, headers=headers).status_code == 200
    assert commerce.read_order(purchase["id"])[1]["status"] == "paid"


def test_unavailable_assets_or_a_changed_total_cannot_reach_payment(store, monkeypatch):
    body = {"dataset": draft(), "email": "buyer@example.test", "expected_amount_cents": 1}
    assert client.post("/api/purchases", json=body).status_code == 409
    monkeypatch.setitem(commerce.CONFIG, "max_archive_bytes", 1)
    assert client.post("/api/purchase-preview", json={"dataset": draft()}).status_code == 422
    assert not (store / "purchases").exists()


def test_demo_is_local_only_and_request_mode_blocks_new_purchases(store, monkeypatch):
    remote = TestClient(app, base_url="https://peelset.com", client=("203.0.113.1", 50000))
    assert remote.post("/api/purchase-preview", json={"dataset": draft()}).status_code == 503
    monkeypatch.setitem(commerce.CONFIG, "checkout_mode", "request")
    assert client.post("/api/purchase-preview", json={"dataset": draft()}).status_code == 409
    original = client.post("/api/dataset-requests", json={**draft(), "email": "buyer@example.test"})
    assert original.status_code == 201 and original.json()["payment_status"] == "not_requested"


def test_demo_shortcut_works_beside_stripe_without_payment_credentials(store, monkeypatch):
    monkeypatch.setitem(commerce.CONFIG, "payment_provider", "stripe")
    monkeypatch.setattr(commerce, "secret", lambda name: None)
    quote = client.post("/api/purchase-preview", json={"dataset": draft()}).json()
    assert quote["payment_provider"] == "stripe" and quote["demo_available"]
    body = {"dataset": draft(), "email": "pitch@example.test", "expected_amount_cents": quote["amount_cents"]}
    assert client.post("/api/purchases", json=body).status_code == 503
    assert not (store / "purchases").exists()
    response = client.post("/api/purchases", json={**body, "demo_checkout": True})
    assert response.status_code == 201, response.text
    purchase = response.json()
    assert purchase["quote"]["payment_provider"] == "demo"
    status = client.get(f"/api/purchases/{purchase['id']}", headers=auth(purchase)).json()
    assert status["status"] == "demo_paid" and status["ready"]
    assert client.get(status["download_url"], headers=auth(purchase)).status_code == 200


def test_demo_shortcut_never_starts_a_stripe_payment(stripe_api):
    purchase = buy({"dataset": draft(), "demo_checkout": True})
    assert "session" not in stripe_api
    assert commerce.read_order(purchase["id"])[1]["status"] == "demo_paid"


def test_demo_shortcut_can_be_disabled_without_disabling_paid_checkout(stripe_api, monkeypatch):
    monkeypatch.setitem(commerce.CONFIG, "demo_checkout_enabled", False)
    quote = client.post("/api/purchase-preview", json={"dataset": draft()}).json()
    assert not quote["demo_available"]
    response = client.post("/api/purchases", json={"dataset": draft(), "demo_checkout": True,
        "email": "buyer@example.test", "expected_amount_cents": quote["amount_cents"]})
    assert response.status_code == 403
    purchase = buy()
    assert purchase["checkout_url"] == "https://checkout.stripe.com/test"
    assert commerce.read_order(purchase["id"])[1]["status"] == "awaiting_payment"


def test_hosted_pitch_requires_explicit_demo_configuration(store, monkeypatch):
    monkeypatch.setitem(commerce.CONFIG, "payment_provider", "stripe")
    monkeypatch.setitem(commerce.CONFIG, "public_url", "https://peelset.com")
    monkeypatch.setattr(commerce, "secret", lambda name: None)
    remote = TestClient(app, base_url="https://peelset.com", client=("203.0.113.1", 50000))
    quote = remote.post("/api/purchase-preview", json={"dataset": draft()}).json()
    assert not quote["demo_available"]
    body = {"dataset": draft(), "demo_checkout": True, "email": "pitch@example.test", "expected_amount_cents": quote["amount_cents"]}
    assert remote.post("/api/purchases", json=body).status_code == 403
    monkeypatch.setitem(commerce.CONFIG, "demo_local_only", False)
    assert remote.post("/api/purchase-preview", json={"dataset": draft()}).json()["demo_available"]
    response = remote.post("/api/purchases", json=body)
    assert response.status_code == 201, response.text
    purchase = response.json()
    assert remote.get(f"/api/purchases/{purchase['id']}/download", headers=auth(purchase)).status_code == 200
