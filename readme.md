# Peelset

A small React + Python website for compiling robotics video datasets. The homepage introduces the banana engine with an editable SVG diagram. `/explore` is one natural-language dataset search, using the original clip explorer design; `/collection` contains bookmarked clips. The working name and domain are **Peelset / peelset.com**.

## Run locally

Requires Node.js 22.12+ and Python 3.9+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. One command runs both servers. Stop them with Ctrl+C. Vite reloads frontend/config edits automatically; the API reads catalog edits on each request.

- Frontend: http://127.0.0.1:5173
- API: http://127.0.0.1:8000/api/clips
- API docs: http://127.0.0.1:8000/docs

Override local ports with `WEB_PORT=5174 API_PORT=8001 npm run dev`.

## Where to edit things

| Change | File |
| --- | --- |
| Brand, domain, browser storage key, and export filename | `config/site.json` |
| Homepage text and engine stage captions | `src/HomePage.jsx` |
| Catalog records, descriptions, labels, image paths, and available variants | `data/catalog.json` |
| Pagination, AI model, variant limits, and request storage | `config/settings.json` |
| Colors and fonts | Variables at the top of `src/styles.css` |
| Explorer and collection text, layout, filters, and clip details | `src/App.jsx` |
| Compiled dataset summary, transformation checklist, and request review text | `src/DatasetPlan.jsx` |
| Banana engine drawings and illustration labels | `src/EngineScene.jsx` |
| Landing page styling | `src/home.css` |
| Metadata search/filter behavior | `backend/catalog.py` |
| AI interpretation, dataset selection, and saved requests | `backend/datasets.py` |
| Browser metadata and favicon | `index.html`, `public/favicon.svg` |

Restart the dev command after changing backend settings. Rebuild for production after changing any frontend file or configuration.

## Catalog behavior

Describe the tasks, objects, and changes you need in the large search box. The compiler proposes source-selection criteria and diffusion transformations. Source filtering uses the actual catalog metadata. Options within a filter group use OR; different groups use AND. Sidebar filters apply in addition to the compiled criteria. Source, upscaled, and augmented tabs filter by versions listed in the source metadata; newly requested generation is shown separately as planned output.

The compiled dataset includes **all matching clips**, across every result page. Use the checkmarks to exclude sources, and use “Adjust changes” to edit object replacements, backgrounds, lighting, resolution, and variants per source. The original prompt travels with the request. Compilation is kept in browser session storage, while sidebar filters and sort order stay in the URL.

“Review dataset” saves a validated request under `.local/dataset-requests/` and offers a JSON download containing the complete source clip IDs, original instructions, generation settings, and planned output counts. These are saved requests, not completed purchases: payment and diffusion generation have not been connected. Request files contain contact information; protect this API with authentication and keep request storage private and persistent before public deployment.

Collections keep bookmarked clip IDs in local storage. You can still save individual clips from their details view and export the collection's metadata.

## Enable live AI interpretation

Without a key, the compiler uses **explicitly labeled demo interpretation** for simple hand-task, pick-and-place, restocking, banana-plantation, tool-use, and lighting examples. Unknown prompts fall back to keyword matching. This is not general AI understanding.

For live interpretation, copy `.env.example` to `.env`, add your `OPENAI_API_KEY`, and restart `npm run dev`. The key stays on the Python server. The dev command reads `.env`; production hosts should supply the key through their environment settings. Model selection defaults to `ai_model` in `config/settings.json`, with an optional `PEELSET_AI_MODEL` override.

The backend sends the prompt and catalog taxonomy to the [OpenAI Responses API with structured output](https://developers.openai.com/api/docs/guides/structured-outputs). It does not send video files. The returned plan is validated, then matched against the actual catalog. Provider errors surface as errors rather than silently switching to demo mode. The generation/payment integration point is at the end of `save_request()` in `backend/datasets.py`.

The 12 included records are **fictional sample metadata**. Images are illustrative stock stills, unrelated to the described actions. No real CCTV, generated video, annotation files, or processing pipeline is connected. To attach a video, add an optional `video_url` to a record; the details view will show a video player. For small local test files, place videos under `public/media/` and use `/media/filename.mp4`. For production footage, connect private storage and access control before adding sensitive data. The current catalog API is public.

Keep each clip's ID unique. `variants` accepts `source`, `upscaled`, and `augmented`. `resolutions` lists the available resolutions; `source_resolution` describes the source. Filter options at the top of `src/App.jsx` must match the values in catalog records. Existing records show the complete format.

## Business context from the remote

Read from `origin/main` (`2eac13e`, the former Relentless landing page) on November 13, 2026:

- Stores get paid for camera footage through store-side hardware, with face blurring and activity filtering at the edge.
- The data engine sorts footage, clips and labels actions, upscales it, and creates diffusion variants. Source frames accompany generated outputs for comparison.
- Customers train robotics, VLA, and world models. Search centers on actions, settings, and objects.
- The remote lists Volume at $4 per footage hour, Refined at $25 per hour, and Custom pricing with bulk physical-drive delivery. These are reference business notes; this starter does not display prices or implement purchasing.

## Deploy later

Build the React frontend and let Python serve both the compiled site and API:

```sh
npm run build
npm start
```

Open http://127.0.0.1:8000. `PORT` controls the production port. Direct visits to `/explore` and `/collection` work in production. `npm run preview` previews the frontend build only; use `npm start` to test the complete production app.

The included multi-stage `Dockerfile` can also run on Coolify (matching the deployment approach in the remote) or another Docker host:

```sh
docker build -t peelset .
docker run --rm -p 8000:8000 peelset
```

Set the service's internal port to **8000**, use `/api/health` as the health-check path, and point `peelset.com` to your host. The domain in the UI is branding only; DNS and live deployment have not been configured. No frontend API URL or CORS setup is needed when Python serves both parts.

## Preview photo sources

Local copies are included so previews do not depend on an image service at runtime:

- Grocery display: https://images.unsplash.com/photo-1515706886582-54c73c5eaf41
- Grocery aisle, Phil Aicken: https://unsplash.com/photos/assorted-bottles-on-display-in-store-o3Qcn8dptnE
- Gas station: https://images.unsplash.com/photo-1583657617216-2b957187f766

Fonts use Google Fonts with local sans-serif fallbacks. Replace the font import in `src/styles.css` to self-host fonts or remove that external request.
