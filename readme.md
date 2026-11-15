# Peelset

A React + Python website for compiling robotics video datasets, and the engine that makes the clips. The homepage introduces the banana engine with an editable SVG diagram. `/explore` is one natural-language dataset search over the real footage library; `/collection` contains saved dataset bundles and bookmarked clips; `/admin` is the Banana Engine console: upload raw footage, watch the engine work through it stage by stage, and publish the clips it makes. The working name and domain are **Peelset / peelset.com**.

```
src/, backend/, config/    the website: React frontend, FastAPI backend
engine/                    the data engine: raw CCTV in, upscaled and labeled hand clips out (see engine/README.md)
data/library/              the footage library that ships with the site: 244 clips, 3 verified object-swap variants
```

## Run locally

Requires Node.js 22.12+ and Python 3.10+ (the pinned Anthropic SDK requires Python 3.10 or newer).

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
| Uploads, engine jobs, publishing, the admin token | `backend/admin.py` |
| The admin console's screens: jobs, the new-job sheet, the job page and its stage views, the dataset | `src/admin/` (`Console.jsx`, `Jobs.jsx`, `Job.jsx`, `Stages.jsx`, `Results.jsx`, `Pages.jsx`) |
| The admin console's look (dark; its own fonts) | `src/admin/console.css` |
| What the engine shows of a stage while it runs | `engine/relentless/live.py` |
| The engine itself: stages, thresholds, models | `engine/` |
| Pagination, AI model, variant limits, and request storage | `config/settings.json` |
| Purchase/request mode, payment provider, pricing, public URL, and download limits | `config/commerce.json` |
| Colors and fonts of the public site | Variables at the top of `src/styles.css` |
| Explorer and collection text, layout, and clip details (video, versions, action segments) | `src/App.jsx`, `src/library.css` |
| Compiled dataset summary and transformation checklist | `src/DatasetPlan.jsx` |
| Dataset review and checkout text | `src/DatasetReview.jsx`, `src/dataset.css` |
| Saved bundles, selection, and checkout summary | `src/Collections.jsx`, `src/collections.css` |
| Banana engine drawings and illustration labels | `src/EngineScene.jsx` |
| Landing page styling | `src/home.css` |
| Metadata search/filter behavior | `backend/catalog.py` |
| AI interpretation, dataset selection, and saved requests | `backend/datasets.py` |
| Saved bundle snapshots and combined checkout requests | `backend/bundles.py` |
| Payment verification, purchase records, and download access | `backend/commerce.py` |
| ZIP contents and archive paths | `backend/delivery.py` |
| Purchase confirmation, downloads, and purchase history text | `src/PurchasePage.jsx`, `src/purchase.css` |
| Demo credit-card entry fields and button text | `src/CardCheckout.jsx`, `src/purchase.css` |
| Browser metadata and favicon | `index.html`, `public/favicon.svg` |

Restart the dev command after changing backend settings. Rebuild for production after changing any frontend file or configuration.

## Catalog behavior

Describe the tasks, objects, and changes you need in the large search box. The compiler proposes source-selection criteria and diffusion transformations. Source filtering uses the actual catalog metadata. Options within a filter group use OR; different groups use AND. Sidebar filters apply in addition to the compiled criteria, and their options and counts come from the catalog (`/api/facets`), so they always match it. Results are ranked: a clip where the wanted action happens on the wanted object comes first. Source, upscaled, and augmented tabs filter by versions listed in the source metadata; newly requested generation is shown separately as planned output.

The compiled dataset includes **all matching clips**, across every result page. Use the checkmarks to exclude sources, and use “Customize video variants” to edit object replacements, backgrounds, lighting, resolution, and variants per source. The original prompt travels with the dataset. Compilation is kept in browser session storage, while sidebar filters and sort order stay in the URL.

“Purchase dataset” opens a review with a server-calculated total. It purchases the **existing library versions**: source clips, available upscaled/object-swap copies, annotations, and provenance. New diffusion transformations are not rendered at checkout; buyers explicitly acknowledge that requested changes are included as instructions only. Video files are deduplicated across bundles, and overlapping source clips are billed once.

“Save to My collections” keeps the entire compiled dataset as one bundle: all selected source IDs across every page, the original instructions, and the transformation settings. The snapshot is stored in `.local/dataset-bundles/`; browser local storage keeps its reference and compact display metadata. Later searches and changes to the catalog do not add sources to an existing bundle. If its sources disappear, checkout asks you to rebuild it. Storage directories, caps, and the checkout bundle limit are in `config/settings.json`.

In **My collections → Dataset bundles**, select several bundles and choose **Checkout selected** to purchase them together. A ZIP is prepared before payment, then the verified purchase unlocks it. The ZIP contains real video and annotation files plus `dataset.json`, which maps source clip IDs to archive paths and preserves each bundle's instructions, plan, license, and attribution. `/purchases` lists purchases made in this browser. The private receipt link also restores access without browser storage; keep it private.

The **Saved clips** tab retains individual bookmarks and metadata export. Both bookmarks and saved bundle references belong to the current browser; no cross-device account sync is connected.

## Checkout settings

Edit `config/commerce.json`. All visible checkout text stays in JSX.

- `checkout_mode: "purchase"` enables purchase and download. Set it to **`"request"`** to restore the original “Your dataset request is saved.” flow and request JSON export. It preserves the current bundle selection and combined-request behavior.
- `demo_card_entry: true` enables the pitch flow: **Purchase dataset → Continue to payment → credit-card entry page → Demo: skip payment → real ZIP download**. The card page is a demo; its fields stay in the browser and are never serialized, stored, or sent to the API. Skip works with blank card fields. The Pay button can also complete a demo using test details. No Stripe credentials are needed for this preview. Set this flag to `false` to restore hosted Stripe Checkout.
- `payment_provider: "stripe"` is the real-payment provider. With `demo_card_entry: false`, the main **Pay & download** button opens Stripe Checkout. Set `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` in `.env` or the deployment environment. Use Stripe test credentials while testing. Set `public_url` to the frontend origin: locally `http://127.0.0.1:5173`, in production `https://peelset.com`. Live keys require HTTPS. Missing credentials block real card checkout while the demo flow remains available.
- `demo_checkout_enabled: true` enables the demo preview and its small **Demo: skip payment** button. It prepares the same real dataset ZIP. Demo purchases are recorded as `demo_paid`, separately from verified Stripe purchases. Set this flag to `false` to disable demo access in both the UI and API. With the card preview off, the skip button appears in the dataset review instead.
- `demo_local_only: true` limits the shortcut and demo downloads to local URLs. For a hosted pitch, set it to `false` and set `public_url` to the deployed origin; visitors can then use the demo shortcut there too. Turn `demo_checkout_enabled` off when the pitch is over. The older `payment_provider: "demo"` mode still works for saved demo orders, but is unnecessary for the pitch shortcut.
- `cents_per_hour: 2500` and `minimum_order_cents: 100` start at **$25 per source hour, $1 minimum**, based on the repo's refined-data business notes. Edit these before taking live payments. Amounts are rounded up to cents and calculated by the server from unique source duration; the browser cannot set the price. Currency currently supports USD.
- `download_days`, `max_archive_bytes`, `max_orders`, and `orders_directory` control download expiry, instant-ZIP size, storage count, and storage location. Oversized datasets need smaller selections or request mode for bulk delivery. Purchase records use SQLite; archives and records live in `.local/purchases/` by default. Keep this folder and bundle storage private and persistent on deployment.

Configure Stripe's webhook endpoint as **`https://peelset.com/api/stripe/webhook`**, listening for `checkout.session.completed`, `checkout.session.async_payment_succeeded`, and `checkout.session.expired`. The backend verifies signatures, retrieves the session from Stripe, checks the session/order ID, amount, currency, live/test mode, and paid status, then records fulfillment once. The receipt page also checks payment on return, so it can unlock promptly while the webhook provides fulfillment if the buyer never returns. This follows [Stripe's fulfillment guidance](https://docs.stripe.com/checkout/fulfillment).

Downloads require a private access token; the browser uses a scoped HttpOnly cookie for the ZIP download. An unpaid order, a success-looking URL, or a client-supplied payment flag cannot unlock an archive. The token lives in the receipt URL's fragment and browser storage, not server request URLs. Private ZIPs are not served by the public media endpoint. Existing catalog previews remain public.

Restart the dev command after editing backend configuration; Vite reloads frontend edits. Rebuild for production. In request mode, saved request JSON lives under `.local/dataset-requests/` and contains contact information. Request mode and purchase mode both leave new diffusion generation unconnected.

## The search box

With `ANTHROPIC_API_KEY` set, the request goes to Claude (`claude_model` in `config/settings.json`, Haiku 5.5 by default; `PEELSET_CLAUDE_MODEL` overrides it) together with the catalog's inventory: its action families, object names, environments and annotations. Claude returns a plan with structured output; values that are not in the catalog are dropped, then the plan is matched against the actual clips. No video is sent. One interpretation costs about 0.05 cents and takes about 4 seconds; the same request is answered from memory the second time, and `ai_calls_per_hour` caps what a public page can spend.

Without a key, or if the AI does not answer, a **keyword interpreter** takes over and the page says so. It knows the action families ("picking up", "handing over"), matches object and place words against the catalog, and understands "replace X with Y" as an object swap. It is not general language understanding. With only `OPENAI_API_KEY` set, the OpenAI Responses API is used instead of Claude (`ai_model`, `PEELSET_AI_MODEL`).

Request mode is handled by `save_request()` in `backend/datasets.py` and `save_checkout()` in `backend/bundles.py`. Purchase mode is handled by `backend/commerce.py`; new diffusion jobs still need a separate generation integration.

## The footage library

Every clip in the catalog is real: a short CCTV crop around one person's hands, with faces blurred, upscaled 4×, with a caption, object names, timed action segments, hand tracks and its lineage back to the source frames. The details view plays the upscaled clip, the original crop, the full scene it was cut from and any verified object-swap variant, and lists the action segments (click one to jump to it).

The catalog is built from two places (`backend/library.py`):

- `data/library/` is the seed library, committed with the site: `catalog.json` plus small streaming copies of each clip. It currently holds 244 clips and 3 variants from 32 five-minute videos of two cameras (a café counter and a snack table), made from the [MEVA dataset](https://mevadata.org) (Kitware Inc. and IARPA, CC BY 4.0).
- `<store>/runs/` holds runs the engine finished on this machine. A run started from the admin console is a private dataset until it is published there ("Publish to catalog"); then its accepted clips and verified variants join the catalog, newest first. `<store>` is `PEELSET_DATA`, or `data/` in the project.

To add clips processed on your own machine to the seed library, run the engine, then export and commit:

```sh
cd engine
uv run python -m relentless.catalog --export ../data/library --web
```

The catalog API and the clip files are public. Only put footage in the library that may be shown publicly.

## Admin: the Banana Engine console

`/admin` asks for the admin token (`ADMIN_TOKEN` on the server; without it the admin API answers 503). Behind it is a dark console, separate from the public site in code and in style (`src/admin/`, loaded only on `/admin`):

- **Jobs**: drop a video on the page to upload it (sent in 8 MB pieces, up to `PEELSET_MAX_UPLOAD_MB`, 2048 by default). A sheet shows the file's length, size and ten small frames, and takes the few choices a job has: the stretch to process, how many clips to refine, the output size, object-swap variants, and where to run it. It estimates time and cost before you press Start. Below are totals over all finished jobs and the list of jobs.
- **A job's page** follows the engine through its eight stages. Each stage has its own live view: the motion filter's scan, person tracking with boxes and skeletons over the footage, busy hands, every hand moment found, Claude's ratings of the contact sheets, each clip before and after upscaling (drag the divider), the label card as it arrives, and object-swap variants with their checks. Beside it are the stage's numbers and the engine's log. Any stage that has run can be opened again by clicking it.
- **When a job is done** the page shows what it made: how much of the footage each stage kept, and the dataset of clips. A clip opens with its labels, action segments (click one to jump to it) and where in the source it comes from. **Publish to catalog** puts the accepted clips, and verified variants, into the public catalog; until then they are private to the console. The menu takes them out again, or deletes the job with its clips. **Export** downloads the labels and lineage as JSON.
- **Overview** shows where jobs can run, free disk space and spend, and warns about what would lose data (no volume) or stop jobs (no key, little memory). **Footage** lists the uploads; one recording can feed any number of jobs. **Datasets** lists what finished jobs made and whether it is published.

Faces are blurred in everything the console shows of a run, by the same rule as in the clips themselves. Two pictures are made before any face has been located, and both are too coarse to show one: the motion filter's view (64 pixels across the frame) and the ten frames of an upload (128 pixels wide each).

A job runs in one of two places:

- **On the server**, if the engine is installed there (`INSTALL_ENGINE=1`): a separate process (`python -m relentless.pipeline --live`), one job at a time. A web server without a GPU is slow at this: a 2-CPU machine needs several minutes per clip. The job only starts if enough memory is free, and it is stopped if the machine runs short, so that other services on the same server are not the ones that suffer.
- **On a worker**: any machine with the engine set up, typically one with a GPU. It asks the site for jobs, downloads the footage, processes it there, keeps the job's page up to date while it works, and sends the finished clips back. On an Apple Silicon Mac that takes about 80 seconds for 25 seconds of footage and four clips at 1080p. Start one with:

  ```sh
  cd engine
  uv run python -m relentless.worker https://peelset.com     # token: --token, or PEELSET_ADMIN_TOKEN in .env
  ```

  The console lists the worker in its sidebar, and new jobs then default to it. A job queued for a worker waits until one connects. The worker uses its own `ANTHROPIC_API_KEY` for labeling. Object-swap variants cost fal credit (about 11 cents each), so they are off unless you turn them on for a job, and only a worker whose `.env` has a `FAL_KEY` can make them; the engine's own cap on fal spend (`FAL_BUDGET_USD`) still applies.

How the page gets its pictures: with `--live` the engine writes previews into the run's `live/` folder (`engine/relentless/live.py`): the frame a stage is on, a small video of the whole stretch, the busiest seconds at full frame rate with everyone's box and keypoints, and stills of the hand moments. The page asks `GET /api/admin/jobs/<id>` about once a second for the run's state, and loads the files from `GET /api/admin/jobs/<id>/file/<path>`. A browser cannot send the token with an `<img>` or a `<video>`, so the page is given a cookie that opens those two read-only file routes and nothing else. The cookie stops working when anyone signs out and when the site restarts (the page then asks for a new one), and signing out also has the browser drop what it cached. A worker's previews are kept in `<store>/live/<job>/` until its finished run arrives.

What a worker sends is read, never trusted (`backend/documents.py`): a worker holds the admin token, but a broken one must not be able to take the site down or publish anything. Its state and catalog are parsed only up to a few megabytes and a plain shape, with NaN read as "no number"; its files must have plain names inside the job's own folders; previews are counted in bytes and in files, per job and in all; a finished run may hold at most 2,000 files. A run is held back from the catalog from the moment its job is made, and what **Publish** puts in the catalog is checked first: a clip whose record is not what the explorer expects, or that names a file outside its own run, is left out.

## Deploy

The multi-stage `Dockerfile` builds the frontend and runs everything in one container, on Coolify or any Docker host:

```sh
docker build --build-arg INSTALL_ENGINE=1 -t peelset .
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=... -e ANTHROPIC_API_KEY=... -v peelset-data:/data -v peelset-orders:/app/.local peelset
```

| Setting | What it does |
| --- | --- |
| `INSTALL_ENGINE` (build argument) | `1` installs the engine with CPU-only PyTorch (image about 2 GB). `0` gives a small image that serves the library and stores uploads but cannot process them. |
| `ADMIN_TOKEN` | Turns on `/admin`. |
| `ANTHROPIC_API_KEY` | AI interpretation of the search box, and labeling in the engine. |
| A volume at `/data` | Keeps uploads, jobs, their datasets and what was published from them across deploys. Without it they live in the container and are lost on the next deploy; the seed library is not affected. |
| A volume at `/app/.local` | Keeps saved bundles, purchase records, private ZIPs, and dataset requests across deploys. Required for durable paid downloads. |
| `RELENTLESS_THREADS` | Caps the CPU threads of an engine job, to leave room for whatever else the server runs. |
| `RELENTLESS_POSE_IMGSZ` | Input size of the pose model. 960 in the image (CPU), 1280 on a Mac. |
| `RELENTLESS_POSE_BATCH` | Frames per pose-model call. 2 in the image, 8 on a Mac. A smaller number needs less memory. |
| `RELENTLESS_LOW_MEMORY` | `1` in the image: person tracking runs in a child process that exits before upscaling starts, which about halves the engine's peak memory. |
| `PEELSET_JOB_MEMORY_MB` | Free memory a job must find before it starts (1200). The job is also stopped if the machine as a whole gets down to `PEELSET_SCARCE_MEMORY_MB` (300), so that it is the job that gives way and not another service on the same server. |
| `PORT` | The port to listen on (8000). |

Set the service's internal port to **8000** and use `/api/health` as the health-check path. Direct visits to `/explore`, `/collection` and `/admin` work in production. No frontend API URL or CORS setup is needed, because Python serves both parts. Without a GPU the engine is slow but works: expect several minutes of processing per minute of footage. On an Apple Silicon Mac it runs 2-3 times faster than real time.

Without Docker: `npm run build`, then `npm start`, and open http://127.0.0.1:8000. `npm run preview` previews the frontend build only.

## Business context from the remote

Read from `origin/main` (`2eac13e`, the former Relentless landing page) on November 13, 2026:

- Stores get paid for camera footage through store-side hardware, with face blurring and activity filtering at the edge.
- The data engine sorts footage, clips and labels actions, upscales it, and creates diffusion variants. Source frames accompany generated outputs for comparison.
- Customers train robotics, VLA, and world models. Search centers on actions, settings, and objects.
- The remote lists Volume at $4 per footage hour, Refined at $25 per hour, and Custom pricing with bulk physical-drive delivery. The purchase starter uses the Refined hourly rate, editable in `config/commerce.json`.

Fonts use Google Fonts with local sans-serif fallbacks. Replace the font import in `src/styles.css` to self-host fonts or remove that external request.
