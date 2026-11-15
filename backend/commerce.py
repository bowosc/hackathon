"""Server-priced checkout, verified Stripe payment, and private dataset downloads."""
import hashlib
import hmac
import json
import logging
import shutil
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from typing import Annotated, List, Optional
from urllib.parse import urlsplit
from uuid import uuid4

import stripe
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from .bundles import BundleId, resolve_bundles
from .catalog import ROOT, SETTINGS, secret
from .datasets import DatasetDraft, DatasetRequest, request_manifest, resolve_selection
from .delivery import archive_plan, write_archive

CONFIG = json.loads((ROOT / "config/commerce.json").read_text())
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
OrderId = Annotated[str, StringConstraints(pattern=r"^order-[a-f0-9]{32}$")]


class PurchaseDraft(BaseModel):
    bundle_ids: List[BundleId] = Field(default_factory=list, max_length=SETTINGS["max_checkout_bundles"])
    dataset: Optional[DatasetDraft] = None


class PurchaseRequest(PurchaseDraft):
    lab: str = Field(default="", max_length=160)
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    accept_available_versions: bool = False
    expected_amount_cents: int = Field(gt=0)
    demo_checkout: bool = False
    demo_card_entry: bool = False


def directory():
    path = (ROOT / CONFIG["orders_directory"]).resolve()
    if not path.is_relative_to(ROOT):
        raise HTTPException(500, "Purchase storage must be inside the project.")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def database():
    connection = sqlite3.connect(directory() / "orders.sqlite3", timeout=10)
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, access_hash TEXT NOT NULL, data TEXT NOT NULL)")
            yield connection
    finally:
        connection.close()


def read_order(order_id):
    with database() as connection:
        row = connection.execute("SELECT access_hash, data FROM orders WHERE id = ?", (order_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Purchase not found.")
    return row[0], json.loads(row[1])


def update_order(order_id, expected_status=None, **changes):
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT data FROM orders WHERE id = ?", (order_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Purchase not found.")
        current = json.loads(row[0])
        if expected_status and current["status"] != expected_status:
            return current
        order = {**current, **changes}
        connection.execute("UPDATE orders SET data = ? WHERE id = ?", (json.dumps(order), order_id))
    return order


def demo_allowed(request):
    local = {"localhost", "127.0.0.1", "::1"}
    return bool(CONFIG.get("demo_checkout_enabled", False) and (not CONFIG.get("demo_local_only", True)
                or (urlsplit(CONFIG["public_url"]).hostname in local and request.url.hostname in local
                    and request.client and request.client.host in local)))


def require_purchase(request):
    if CONFIG["checkout_mode"] != "purchase":
        raise HTTPException(409, "This site is currently using dataset request mode.")
    if CONFIG["payment_provider"] not in {"stripe", "demo"}:
        raise HTTPException(500, "Configure a supported payment provider.")
    if CONFIG["payment_provider"] == "demo" and not demo_allowed(request):
        raise HTTPException(503, "Demo checkout is only available locally. Configure Stripe for the live site.")
    base = urlsplit(CONFIG["public_url"])
    if base.scheme not in {"http", "https"} or not base.netloc or base.query or base.fragment:
        raise HTTPException(500, "Configure a valid public_url for checkout.")


def resolve_purchase(draft):
    if bool(draft.bundle_ids) == bool(draft.dataset):
        raise HTTPException(422, "Select saved bundles or one compiled dataset.")
    if draft.bundle_ids:
        return [(bundle, clips) for bundle, clips in resolve_bundles(draft.bundle_ids)]
    _, clips = resolve_selection(draft.dataset.selection)
    # Share the same validation as the original request flow.
    request_manifest(DatasetRequest(**draft.dataset.model_dump(), email="preview@example.invalid"), clips)
    if not draft.dataset.prompt.strip():
        raise HTTPException(422, "Enter dataset instructions.")
    return [(draft.dataset, clips)]


def price(resolved, package):
    unique = {clip["id"]: clip for _, clips in resolved for clip in clips}
    seconds = sum(Decimal(str(clip["duration_seconds"])) for clip in unique.values())
    amount = max(CONFIG["minimum_order_cents"], int((seconds * CONFIG["cents_per_hour"] / 3600).to_integral_value(rounding=ROUND_CEILING)))
    if amount < 50 or CONFIG["cents_per_hour"] <= 0 or CONFIG["currency"] != "usd":
        raise HTTPException(500, "Configure valid USD dataset pricing.")
    return {"amount_cents": amount, "currency": CONFIG["currency"], "cents_per_hour": CONFIG["cents_per_hour"],
            "minimum_order_cents": CONFIG["minimum_order_cents"], "bundle_count": len(resolved),
            "clip_count": sum(len(clips) for _, clips in resolved), "unique_clip_count": len(unique),
            "duration_seconds": float(seconds), "video_count": package["video_count"], "bytes": package["bytes"],
            "requires_custom_ack": any(draft.selection.plan.modifications for draft, _ in resolved),
            "payment_provider": CONFIG["payment_provider"]}


def prepare(draft):
    resolved = resolve_purchase(draft)
    unique = {clip["id"]: clip for _, clips in resolved for clip in clips}
    package = archive_plan(list(unique.values()), CONFIG["max_archive_bytes"])
    return resolved, package, price(resolved, package)


@router.post("/purchase-preview")
def preview(draft: PurchaseDraft, request: Request):
    require_purchase(request)
    return {**prepare(draft)[2], "demo_available": demo_allowed(request)}


def stripe_call(method, *args, **kwargs):
    try:
        return method(*args, api_key=secret("STRIPE_SECRET_KEY"), **kwargs)
    except stripe.StripeError:
        raise HTTPException(503, "The payment service could not be reached. Please try again.")


@router.post("/purchases", status_code=201)
def create_purchase(buyer: PurchaseRequest, request: Request):
    require_purchase(request)
    if buyer.demo_card_entry and not CONFIG.get("demo_card_entry", False):
        raise HTTPException(403, "The demo payment screen is not enabled here.")
    provider = "demo" if buyer.demo_checkout or buyer.demo_card_entry else CONFIG["payment_provider"]
    if provider == "demo":
        if not demo_allowed(request):
            raise HTTPException(403, "Demo checkout is not enabled here.")
    else:
        key = secret("STRIPE_SECRET_KEY")
        if not key or not secret("STRIPE_WEBHOOK_SECRET"):
            logger.warning("Card checkout requires STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET.")
            raise HTTPException(503, {"code": "checkout_unavailable"})
        if key.startswith("sk_live_") and urlsplit(CONFIG["public_url"]).scheme != "https":
            logger.warning("Live checkout requires an HTTPS public_url.")
            raise HTTPException(503, {"code": "checkout_unavailable"})
    resolved, package, quote = prepare(buyer)
    quote["payment_provider"] = provider
    if buyer.expected_amount_cents != quote["amount_cents"]:
        raise HTTPException(409, "The dataset total has changed. Refresh the total and review it before paying.")
    if quote["requires_custom_ack"] and not buyer.accept_available_versions:
        raise HTTPException(422, "Confirm that this purchase is for the available video versions.")
    order_id, token = f"order-{uuid4().hex}", secrets.token_urlsafe(32)
    now = time.time()
    bundles = []
    for draft, clips in resolved:
        item = DatasetRequest(prompt=draft.prompt, interpretation_mode=draft.interpretation_mode,
                              selection=draft.selection, lab=buyer.lab, email=buyer.email)
        details = request_manifest(item, clips)
        bundles.append({**{key: details[key] for key in ["source_clip_ids", "source_duration_seconds", "generation_spec"]},
                        "bundle_id": getattr(draft, "id", None),
                        "bundle_name": getattr(draft, "name", draft.prompt[:120])})
    manifest = {"id": order_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "fulfillment": "available_library", "requested_transformations_status": "not_generated",
                "bundles": bundles, "quote": quote}
    order = {"id": order_id, "status": "awaiting_payment", "provider": provider, "quote": quote,
             "buyer": {"lab": buyer.lab, "email": buyer.email}, "archive_ready": False,
             "created_at": now, "expires_at": now + CONFIG["download_days"] * 86400,
             "session_id": None, "checkout_url": None, "manifest": manifest}
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] >= CONFIG["max_orders"]:
            raise HTTPException(503, "Purchase storage is full. Please try again later.")
        # Reserve space for concurrent packages, leaving the machine a little breathing room.
        reserved = connection.execute("SELECT COALESCE(SUM(json_extract(data, '$.quote.bytes')), 0) FROM orders WHERE json_extract(data, '$.archive_ready') = 0").fetchone()[0]
        if shutil.disk_usage(directory()).free < reserved + package["bytes"] + 1024 * 1024 * 1024:
            raise HTTPException(503, "There is not enough space to prepare this download. Please try again later.")
        connection.execute("INSERT INTO orders VALUES (?, ?, ?)",
                           (order_id, hashlib.sha256(token.encode()).hexdigest(), json.dumps(order)))
    path = directory() / f"{order_id}.zip"
    receipt = f"{CONFIG['public_url'].rstrip('/')}/purchases/{order_id}#access={token}"
    try:
        write_archive(path, manifest, package)   # prepare every promised file before accepting payment
        update_order(order_id, archive_ready=True)
        if order["provider"] == "stripe":
            session = stripe_call(stripe.checkout.Session.create, mode="payment", payment_method_types=["card"],
                client_reference_id=order_id, customer_email=buyer.email, metadata={"order_id": order_id},
                line_items=[{"price_data": {"currency": quote["currency"], "unit_amount": quote["amount_cents"],
                    "product_data": {"name": "Peelset video dataset", "description": f"{quote['unique_clip_count']} clips, {quote['video_count']} available video files and annotations"}}, "quantity": 1}],
                success_url=receipt, cancel_url=receipt, expires_at=int(now + 1800), idempotency_key=order_id)
            update_order(order_id, session_id=session.id, checkout_url=session.url, livemode=session.livemode)
            checkout_url = session.url
        else:
            if buyer.demo_checkout:
                update_order(order_id, status="demo_paid")
            checkout_url = receipt
    except Exception:
        path.unlink(missing_ok=True)
        with database() as connection:
            connection.execute("DELETE FROM orders WHERE id = ?", (order_id,))
        raise
    return {"id": order_id, "access_token": token, "checkout_url": checkout_url, "receipt_url": receipt, "quote": quote}


def authorized(order_id, request):
    access_hash, order = read_order(order_id)
    bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
    token = bearer or request.cookies.get("peelset_download", "")
    if not token or not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), access_hash):
        raise HTTPException(404, "Open the private link saved after checkout to access this purchase.")
    if order["provider"] == "demo" and not demo_allowed(request):
        raise HTTPException(403, "Demo purchases are only available locally.")
    return order, token


def fulfill(session_id):
    session = stripe_call(stripe.checkout.Session.retrieve, session_id)
    order_id = getattr(getattr(session, "metadata", None), "order_id", None)
    if not order_id:
        return
    _, order = read_order(order_id)
    if (order["provider"] != "stripe" or order["session_id"] != session.id or session.client_reference_id != order_id
        or session.amount_total != order["quote"]["amount_cents"] or session.currency != order["quote"]["currency"]
        or session.livemode != order.get("livemode")):
        raise HTTPException(409, "The payment does not match this purchase.")
    if session.payment_status == "paid" and session.status == "complete":
        if order["status"] != "paid":
            update_order(order_id, expected_status="awaiting_payment", status="paid", paid_at=time.time(), expires_at=time.time() + CONFIG["download_days"] * 86400)
    elif session.status == "expired" and order["status"] != "paid":
        update_order(order_id, expected_status="awaiting_payment", status="expired")


@router.get("/purchases/{order_id}")
def purchase_status(order_id: OrderId, request: Request, response: Response):
    order, token = authorized(order_id, request)
    if order["provider"] == "stripe" and order["status"] == "awaiting_payment" and order["session_id"]:
        fulfill(order["session_id"])
        _, order = read_order(order_id)
    ready = order["status"] in {"paid", "demo_paid"} and order["expires_at"] > time.time()
    response.set_cookie("peelset_download", token, path=f"/api/purchases/{order_id}", httponly=True,
                        secure=request.url.scheme == "https", samesite="strict", max_age=86400)
    response.headers.update({"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
    return {key: order[key] for key in ["id", "status", "provider", "quote", "expires_at", "checkout_url"]} | {
        "ready": ready, "download_url": f"/api/purchases/{order_id}/download" if ready else None,
        "bundles": [{"name": item["bundle_name"], "clip_count": len(item["source_clip_ids"])} for item in order["manifest"]["bundles"]]}


@router.post("/purchases/{order_id}/demo-pay")
def demo_pay(order_id: OrderId, request: Request):
    order, _ = authorized(order_id, request)
    if order["provider"] != "demo" or not demo_allowed(request):
        raise HTTPException(403, "This is not a local demo checkout.")
    if order["status"] == "awaiting_payment":
        update_order(order_id, expected_status="awaiting_payment", status="demo_paid", expires_at=time.time() + CONFIG["download_days"] * 86400)
    return {"status": "demo_paid"}


@router.get("/purchases/{order_id}/download")
def download_purchase(order_id: OrderId, request: Request):
    order, _ = authorized(order_id, request)
    if order["status"] not in {"paid", "demo_paid"}:
        raise HTTPException(402, "Complete payment before downloading this dataset.")
    if order["expires_at"] <= time.time():
        raise HTTPException(410, "This download link has expired.")
    path = directory() / f"{order_id}.zip"
    if not path.is_file():
        raise HTTPException(503, "Your download is temporarily unavailable. Please contact Peelset with your purchase ID.")
    return FileResponse(path, filename=f"peelset-{order_id}.zip", media_type="application/zip",
                        headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer"})


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    signing_secret = secret("STRIPE_WEBHOOK_SECRET")
    if not signing_secret:
        raise HTTPException(503, "Webhook is not configured.")
    try:
        event = stripe.Webhook.construct_event(await request.body(), request.headers.get("stripe-signature", ""), signing_secret)
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature.")
    if event.type in {"checkout.session.completed", "checkout.session.async_payment_succeeded", "checkout.session.expired"}:
        await run_in_threadpool(fulfill, event.data.object.id)
    return {"received": True}
