"""Saved bundle snapshots and a single request for multiple bundles."""
from datetime import datetime, timezone
from typing import Annotated, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from .catalog import ROOT, SETTINGS, catalog
from .datasets import DatasetDraft, DatasetRequest, Name, persist_request, request_manifest, resolve_selection

router = APIRouter(prefix="/api")
BundleId = Annotated[str, StringConstraints(pattern=r"^bundle-[a-f0-9]{32}$")]


class BundleDraft(DatasetDraft):
    name: str = Field(default="", max_length=120)


class StoredBundle(BundleDraft):
    id: BundleId
    created_at: str
    source_clip_ids: List[Name]


class BundleSelection(BaseModel):
    bundle_ids: List[BundleId] = Field(min_length=1, max_length=SETTINGS["max_checkout_bundles"])


class CheckoutRequest(BundleSelection):
    lab: str = Field(default="", max_length=160)
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def bundle_directory():
    directory = (ROOT / SETTINGS["bundles_directory"]).resolve()
    if not directory.is_relative_to(ROOT):
        raise HTTPException(500, "Bundle storage must be inside the project.")
    return directory


def describe(bundle, clips):
    return {"id": bundle.id, "name": bundle.name, "created_at": bundle.created_at,
            "prompt": bundle.prompt, "plan": bundle.selection.plan.model_dump(),
            "clip_count": len(clips), "duration_seconds": sum(clip["duration_seconds"] for clip in clips),
            "previews": [{key: clip.get(key) for key in ["id", "image", "image_position"]} for clip in clips[:3]]}


@router.post("/bundles", status_code=201)
def save_bundle(draft: BundleDraft):
    if not draft.prompt.strip() or draft.selection.plan is None:
        raise HTTPException(422, "Compile a dataset before saving a bundle.")
    if "object_swap" in draft.selection.plan.modifications and not draft.selection.plan.target_object.strip():
        raise HTTPException(422, "Enter a target object before saving this bundle.")
    _, clips = resolve_selection(draft.selection)
    if not clips:
        raise HTTPException(422, "Select at least one source clip.")
    bundle = StoredBundle(**{**draft.model_dump(), "name": draft.name.strip() or draft.prompt.strip()[:120]},
                          id=f"bundle-{uuid4().hex}", created_at=datetime.now(timezone.utc).isoformat(),
                          source_clip_ids=[clip["id"] for clip in clips])
    directory = bundle_directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if sum(1 for _ in directory.iterdir()) >= SETTINGS["max_saved_bundles"]:
            raise HTTPException(503, "The bundle storage is full. Please try again later.")
        (directory / f"{bundle.id}.json").write_text(bundle.model_dump_json(indent=2))
    except OSError:
        raise HTTPException(503, "The bundle could not be saved. Please try again.")
    return describe(bundle, clips)


def resolve_bundles(bundle_ids):
    if len(set(bundle_ids)) != len(bundle_ids):
        raise HTTPException(422, "Select each bundle once.")
    by_id = {clip["id"]: clip for clip in catalog()}
    result = []
    for bundle_id in bundle_ids:
        try:
            bundle = StoredBundle.model_validate_json((bundle_directory() / f"{bundle_id}.json").read_text())
        except (OSError, ValueError):
            raise HTTPException(409, "A saved bundle is no longer available. Remove it and save it again from the explorer.")
        if bundle.selection.plan is None or not bundle.source_clip_ids or set(bundle.source_clip_ids) - by_id.keys():
            raise HTTPException(409, f"Some source clips in ‘{bundle.name}’ are no longer available. Rebuild that bundle.")
        result.append((bundle, [by_id[clip_id] for clip_id in bundle.source_clip_ids]))
    return result


def checkout_description(resolved):
    all_clips = [clip for _, clips in resolved for clip in clips]
    unique = {clip["id"]: clip for clip in all_clips}
    return {"bundles": [describe(bundle, clips) for bundle, clips in resolved],
            "totals": {"bundle_count": len(resolved), "clip_count": len(all_clips), "unique_clip_count": len(unique),
                       "duration_seconds": sum(clip["duration_seconds"] for clip in all_clips)}}


@router.post("/bundle-checkout-preview")
def checkout_preview(selection: BundleSelection):
    return checkout_description(resolve_bundles(selection.bundle_ids))


@router.post("/dataset-checkouts", status_code=201)
def save_checkout(request: CheckoutRequest):
    resolved = resolve_bundles(request.bundle_ids)
    overview = checkout_description(resolved)
    manifests = []
    for bundle, clips in resolved:
        item = DatasetRequest(prompt=bundle.prompt, interpretation_mode=bundle.interpretation_mode,
                              selection=bundle.selection, lab=request.lab, email=request.email)
        manifests.append({**request_manifest(item, clips), "bundle_id": bundle.id, "bundle_name": bundle.name})
    manifest = {"id": f"PS-CHECKOUT-{uuid4().hex}", "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "saved", "generation_status": "not_started", "payment_status": "not_requested",
                "buyer": {"lab": request.lab, "email": request.email}, "totals": overview["totals"], "bundles": manifests}
    persist_request(manifest)
    # Pricing/payment and generation can use this one checkout and its separate bundle specifications.
    return manifest
