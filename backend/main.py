"""Catalog and bulk dataset API. Edit config/ and data/ to change the demo."""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from . import library
from .admin import MAX_CHUNK, media_router
from .admin import router as admin_router
from .catalog import ROOT, SETTINGS, catalog, filter_clips
from .datasets import router as datasets_router

app = FastAPI(title="Peelset catalog", version="0.2.0")


class BodyTooLarge(Exception):
    pass


class BodyLimit:
    """Refuse a request body that is larger than its route can need, before anything holds it in memory.

    JSON requests get 1 MB. Pieces of an upload (PUT under /api/admin) get the largest piece plus some room.
    Without this, anyone could post hundreds of megabytes at a small server: the body of a JSON request is
    read in full before the route, or its admin token check, gets a say."""

    def __init__(self, inner, small=1024 * 1024, piece=MAX_CHUNK + 1024 * 1024):
        self.inner, self.small, self.piece = inner, small, piece

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        limit = self.piece if scope["method"] == "PUT" and scope["path"].startswith("/api/admin/") else self.small
        declared = dict(scope["headers"]).get(b"content-length", b"")
        arrived, over, answered = 0, declared.isdigit() and int(declared) > limit, False

        async def counted():
            nonlocal arrived, over
            message = await receive()
            if message["type"] == "http.request":
                arrived += len(message.get("body") or b"")
                if arrived > limit:   # no length was declared, or it was wrong
                    over = True
                    raise BodyTooLarge
            return message

        async def tracked(message):
            nonlocal answered
            if not over:   # what the app makes of a read that was cut short is replaced by the answer below
                answered = answered or message["type"] == "http.response.start"
                await send(message)

        try:
            if not over:
                await self.inner(scope, counted, tracked)
        except BodyTooLarge:
            pass
        if over and not answered:
            body = b'{"detail":"That request is too large."}'
            await send({"type": "http.response.start", "status": 413, "headers": [
                (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})


app.add_middleware(BodyLimit)
app.include_router(datasets_router)
app.include_router(admin_router)
app.include_router(media_router)


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


@app.get("/api/facets")
def facets():
    """The explorer's filter groups, with how many clips each option holds."""
    return {"groups": library.facets()}


@app.get("/api/media/{place}/{path:path}")
def media(place: str, path: str):
    """Clip videos, posters and label files, from the seed library or from runs processed here."""
    file = library.media_file(place, path)
    if file is None:
        raise HTTPException(404, "File not found")
    return FileResponse(file, headers={"Cache-Control": "public, max-age=3600"})


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
