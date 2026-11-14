# Peelset

A React + Python website for compiling robotics video datasets, and the engine that makes the clips. The homepage introduces the banana engine with an editable SVG diagram. `/explore` is one natural-language dataset search over the real footage library; `/collection` contains bookmarked clips; `/admin` uploads raw footage and runs the engine on it. The working name and domain are **Peelset / peelset.com**.

```
src/, backend/, config/    the website: React frontend, FastAPI backend
engine/                    the data engine: raw CCTV in, upscaled and labeled hand clips out (see engine/README.md)
data/library/              the footage library that ships with the site: 34 clips, 3 verified object-swap variants
```

## Run locally

Requires Node.js 22.12+ and Python 3.9+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. One command runs both servers. Stop them with Ctrl+C. Vite reloads frontend/config edits automatically; the API picks up catalog changes on the next request.

That is enough to browse and search the library. Two optional things, both set in `.env` (copy `.env.example`):

- `ANTHROPIC_API_KEY` turns on the AI interpreter for the search box and lets the engine label clips.
- `ADMIN_TOKEN` turns on `/admin`. To process footage on your machine, also set up the engine once: `cd engine && uv sync` (see `engine/README.md`). The site finds `engine/.venv` on its own.

Tests: `.venv/bin/python -m pip install -r requirements-dev.txt`, then `.venv/bin/python -m pytest tests`. The engine has its own: `cd engine && uv run pytest`.

- Frontend: http://127.0.0.1:5173
- API: http://127.0.0.1:8000/api/clips
- API docs: http://127.0.0.1:8000/docs

Override local ports with `WEB_PORT=5174 API_PORT=8001 npm run dev`.

## Where to edit things

| Change | File |
| --- | --- |
| Brand, domain, browser storage key, and export filename | `config/site.json` |
| Homepage text and engine stage captions | `src/HomePage.jsx` |
| The clips in the catalog | `data/library/` (seed library) and the engine's runs; see "The footage library" below |
| How an engine clip becomes a catalog record: title, action families, environment, lighting, filter groups | `backend/library.py` |
| Uploads, engine jobs, the admin token | `backend/admin.py`, `src/Admin.jsx`, `src/admin.css` |
| The engine itself: stages, thresholds, models | `engine/` |
| Pagination, AI model, variant limits, and request storage | `config/settings.json` |
| Colors and fonts | Variables at the top of `src/styles.css` |
| Explorer and collection text, layout, and clip details (video, versions, action segments) | `src/App.jsx`, `src/library.css` |
| Compiled dataset summary, transformation checklist, and request review text | `src/DatasetPlan.jsx` |
| Banana engine drawings and illustration labels | `src/EngineScene.jsx` |
| Landing page styling | `src/home.css` |
| Metadata search/filter behavior | `backend/catalog.py` |
| AI interpretation, dataset selection, and saved requests | `backend/datasets.py` |
| Browser metadata and favicon | `index.html`, `public/favicon.svg` |

Restart the dev command after changing backend settings. Rebuild for production after changing any frontend file or configuration.

## Catalog behavior

Describe the tasks, objects, and changes you need in the large search box. The compiler proposes source-selection criteria and diffusion transformations. Source filtering uses the actual catalog metadata. Options within a filter group use OR; different groups use AND. Sidebar filters apply in addition to the compiled criteria, and their options and counts come from the catalog (`/api/facets`), so they always match it. Results are ranked: a clip where the wanted action happens on the wanted object comes first. Source, upscaled, and augmented tabs filter by versions listed in the source metadata; newly requested generation is shown separately as planned output.

The compiled dataset includes **all matching clips**, across every result page. Use the checkmarks to exclude sources, and use “Adjust changes” to edit object replacements, backgrounds, lighting, resolution, and variants per source. The original prompt travels with the request. Compilation is kept in browser session storage, while sidebar filters and sort order stay in the URL.

“Review dataset” saves a validated request under `.local/dataset-requests/` and offers a JSON download containing the complete source clip IDs, original instructions, generation settings, and planned output counts. These are saved requests, not completed purchases: payment and diffusion generation have not been connected. Request files contain contact information; protect this API with authentication and keep request storage private and persistent before public deployment.

Collections keep bookmarked clip IDs in local storage. You can still save individual clips from their details view and export the collection's metadata.

## The search box

With `ANTHROPIC_API_KEY` set, the request goes to Claude (`claude_model` in `config/settings.json`, Haiku 5.5 by default; `PEELSET_CLAUDE_MODEL` overrides it) together with the catalog's inventory: its action families, object names, environments and annotations. Claude returns a plan with structured output; values that are not in the catalog are dropped, then the plan is matched against the actual clips. No video is sent. One interpretation costs about 0.05 cents and takes about 4 seconds; the same request is answered from memory the second time, and `ai_calls_per_hour` caps what a public page can spend.

Without a key, or if the AI does not answer, a **keyword interpreter** takes over and the page says so. It knows the action families ("picking up", "handing over"), matches object and place words against the catalog, and understands "replace X with Y" as an object swap. It is not general language understanding. With only `OPENAI_API_KEY` set, the OpenAI Responses API is used instead of Claude (`ai_model`, `PEELSET_AI_MODEL`).

The generation/payment integration point is at the end of `save_request()` in `backend/datasets.py`.

## The footage library

Every clip in the catalog is real: a short CCTV crop around one person's hands, with faces blurred, upscaled 4×, with a caption, object names, timed action segments, hand tracks and its lineage back to the source frames. The details view plays the upscaled clip, the original crop, the full scene it was cut from and any verified object-swap variant, and lists the action segments (click one to jump to it).

The catalog is built from two places (`backend/library.py`):

- `data/library/` is the seed library, committed with the site: `catalog.json` plus small streaming copies of each clip. It currently holds 34 clips and 3 variants made from the [MEVA dataset](https://mevadata.org) (Kitware Inc. and IARPA, CC BY 4.0).
- `<store>/runs/` holds runs the engine finished on this machine, for example from an admin upload. Their accepted clips join the catalog as soon as the run ends. `<store>` is `PEELSET_DATA`, or `data/` in the project.

To add clips processed on your own machine to the seed library, run the engine, then export and commit:

```sh
cd engine
uv run python -m relentless.catalog --export ../data/library --web
```

The catalog API and the clip files are public. Only put footage in the library that may be shown publicly.

## Admin: upload footage and run the engine

`/admin` asks for the admin token (`ADMIN_TOKEN` on the server; without it the admin API answers 503). It shows whether the engine is installed, free disk space and what labeling has cost, takes video uploads (sent in 8 MB pieces, up to `PEELSET_MAX_UPLOAD_MB`, 2048 by default), and starts engine jobs: pick the start, the length, how many clips to refine and the output size. Jobs run one at a time as a separate process (`python -m relentless.pipeline`), the page follows the stages live, and removing a finished job also removes its clips from the catalog. Each refined clip costs about 0.2 cents of Claude; object-swap variants (fal) are not started from this page.

## Deploy

The multi-stage `Dockerfile` builds the frontend and runs everything in one container, on Coolify or any Docker host:

```sh
docker build --build-arg INSTALL_ENGINE=1 -t peelset .
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=... -e ANTHROPIC_API_KEY=... -v peelset-data:/data peelset
```

| Setting | What it does |
| --- | --- |
| `INSTALL_ENGINE` (build argument) | `1` installs the engine with CPU-only PyTorch (image about 2 GB). `0` gives a small image that serves the library and stores uploads but cannot process them. |
| `ADMIN_TOKEN` | Turns on `/admin`. |
| `ANTHROPIC_API_KEY` | AI interpretation of the search box, and labeling in the engine. |
| A volume at `/data` | Keeps uploads, runs and the job list across deploys. Without it they live in the container and are lost on the next deploy; the seed library is not affected. |
| `RELENTLESS_THREADS` | Caps the CPU threads of an engine job, to leave room for whatever else the server runs. |
| `RELENTLESS_POSE_IMGSZ` | Input size of the pose model. 960 in the image (CPU), 1280 on a Mac. |
| `PORT` | The port to listen on (8000). |

Set the service's internal port to **8000** and use `/api/health` as the health-check path. Direct visits to `/explore`, `/collection` and `/admin` work in production. No frontend API URL or CORS setup is needed, because Python serves both parts. Without a GPU the engine is slow but works: expect several minutes of processing per minute of footage. On an Apple Silicon Mac it runs 2-3 times faster than real time.

Without Docker: `npm run build`, then `npm start`, and open http://127.0.0.1:8000. `npm run preview` previews the frontend build only.

## Business context from the remote

Read from `origin/main` (`2eac13e`, the former Relentless landing page) on November 13, 2026:

- Stores get paid for camera footage through store-side hardware, with face blurring and activity filtering at the edge.
- The data engine sorts footage, clips and labels actions, upscales it, and creates diffusion variants. Source frames accompany generated outputs for comparison.
- Customers train robotics, VLA, and world models. Search centers on actions, settings, and objects.
- The remote lists Volume at $4 per footage hour, Refined at $25 per hour, and Custom pricing with bulk physical-drive delivery. These are reference business notes; this starter does not display prices or implement purchasing.

Fonts use Google Fonts with local sans-serif fallbacks. Replace the font import in `src/styles.css` to self-host fonts or remove that external request.
