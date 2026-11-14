"""Catalog and bulk dataset API. Edit config/ and data/ to change the demo."""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

from .catalog import ROOT, SETTINGS, catalog, filter_clips
from .datasets import router as datasets_router

app = FastAPI(title="Peelset catalog", version="0.1.0")
app.include_router(datasets_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/clips")
def list_clips(
    q: str = Query("", max_length=300),
    environment: List[str] = Query([]),
    action: List[str] = Query([]),
    annotation: List[str] = Query([]),
    resolution: List[str] = Query([]),
    lighting: List[str] = Query([]),
    variant: Optional[str] = Query(None, pattern="^(source|upscaled|augmented)$"),
    sort: str = Query(SETTINGS["default_sort"], pattern="^(relevance|newest|duration)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(SETTINGS["page_size"], ge=1, le=SETTINGS["max_page_size"]),
):
    filters = {"environment": environment, "actions": action, "annotations": annotation,
               "resolutions": resolution, "lighting": lighting}
    clips = catalog()
    matches = filter_clips(clips, q, filters, variant, sort)
    return {"clips": matches[offset:offset + limit], "total": len(matches),
            "catalog_total": len(clips), "offset": offset, "limit": limit}


@app.get("/api/collection")
def collection(ids: List[str] = Query([])):
    by_id = {clip["id"]: clip for clip in catalog()}
    return {"clips": [by_id[clip_id] for clip_id in dict.fromkeys(ids) if clip_id in by_id]}


@app.get("/api/clips/{clip_id}")
def clip_detail(clip_id: str):
    for clip in catalog():
        if clip["id"] == clip_id:
            return clip
    raise HTTPException(404, "Clip not found")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def missing_api(path: str):
    raise HTTPException(404, "API endpoint not found")


# After `npm run build`, Python can serve the whole app in production.
class ReactFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code == 404 and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


if (ROOT / "dist").is_dir():
    app.mount("/", ReactFiles(directory=ROOT / "dist", html=True), name="frontend")
