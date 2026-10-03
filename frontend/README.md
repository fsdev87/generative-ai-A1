# GenAI Studio frontend

React 19 + TypeScript + Tailwind CSS v3, built with Vite. It is the browser application for the
four workspaces: **Universal Restoration**, **Hard-Routed Restoration**, **Soft
Mixture-of-Experts Restoration** and **Face-to-Sketch Generator**. Every number it shows comes
from the FastAPI backend (`docs/api.md`); nothing is simulated in the browser.

## Running

### With Docker (what the evaluator uses)

From the repository root:

```bash
docker compose up --build
```

Open <http://localhost:5173>. The frontend container (nginx) serves the built app and forwards
`/api/*` to the backend container, so the browser only talks to one origin.

### Development

Requires Node 22 and a running backend on <http://localhost:8000> (for example
`docker compose up backend`, or `python -m uvicorn app.main:app --app-dir backend --port 8000`).

```bash
cd frontend
npm ci              # once
npm run dev         # http://localhost:5173, /api is proxied to the backend
```

Set `BACKEND_URL` to proxy to a different backend (`BACKEND_URL=http://localhost:9000 npm run dev`).

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload and the `/api` proxy |
| `npm run typecheck` | TypeScript type-check (`tsc --noEmit`, strict mode) |
| `npm run build` | Type-check and production build into `dist/` |
| `npm test` | Unit and component tests (Vitest + Testing Library, jsdom, mocked `fetch`) |
| `npm run preview` | Serve `dist/` locally with the same `/api` proxy |

## Structure

```
frontend/
  Dockerfile, nginx.conf, .dockerignore   production image: node build stage, then nginx
  index.html, vite.config.ts              Vite entry, dev proxy, Vitest settings
  tailwind.config.ts, postcss.config.js   Stitch design tokens
  src/
    main.tsx, App.tsx, index.css          entry point, app shell (top bar, workspaces, footer)
    api/types.ts                          TypeScript mirror of backend/app/schemas.py
    api/client.ts                         fetch wrapper, ApiError, multipart form builders
    hooks/backend.tsx                     health polling, options and samples (React context)
    hooks/useRequest.ts                   one request at a time: loading, result, error, cancel
    hooks/useSelectedImage.ts             chosen upload / webcam frame / sample + preview URL
    lib/                                  formatting, routes, download names, error map, camera
    components/                           shared design-system components (see below)
    workspaces/                           the four workspaces
    workspaces/restoration/               input card, image panels and logic shared by Tasks 1-3
    test/                                 test setup and API fixtures
```

All four workspaces stay mounted and are hidden when inactive, so inputs and results survive
switching tabs. Each workspace has its own URL (`/universal-restoration`,
`/hard-routed-restoration`, `/soft-mixture-of-experts-restoration`,
`/face-to-sketch-generator`; short forms `/universal`, `/hard`, `/moe`, `/sketch`).

## How the Stitch design maps to the code

The Google Stitch export is in `design/stitch/` (one HTML file per workspace). Its Tailwind
config (Material 3 colour roles, radii, spacing, Inter type scale) is copied unchanged into
`tailwind.config.ts`, and the components reuse the export's class names. Fonts (Inter) and icons
(Material Symbols Outlined) are bundled from npm packages instead of Google's CDN, and Tailwind is
compiled at build time instead of `cdn.tailwindcss.com`, so the app works offline.

| Stitch element | Component |
|---|---|
| Header: logo, tabs, "Backend online" pill, avatar | `TopBar` (`StatusPill`; the avatar became the System information button) |
| Workspace title row with chips | `WorkspaceHeader` (chips show real model files and ONNX Runtime) |
| 12-column grid, input left (4), results right (8) | `WorkspaceLayout` |
| "Input" card: dropzone, sample row, Corruption / Severity segmented controls, Restore | `RestorationInputCard` with `Dropzone`, `SamplePicker`, `SegmentedControl` |
| Routing Predicted / Oracle pill switch | `RoutingToggle` (in `HardRoutedWorkspace.tsx`) |
| Original / Model input / Restored output panels | `RestorationPanels` with `ImagePanel` |
| Classifier card with probability bars and badges | `ClassifierCard` with `ScoreBars` (grid variant) |
| Expert weights card, "Main contributor", branch thumbnails | `ExpertWeightsCard` with `ScoreBars` (list variant) |
| Compact details card + Download button | `DetailsCard`, `DownloadButton` |
| "Reconstruction Delta Map" card | `ErrorMapCard` (error map computed in the browser) |
| Sketch "Photo" card: dropzone, Use webcam, Sketch style, Generate | `SketchWorkspace` with `WebcamCapture` |
| Photo / Generated sketch panels | `ImagePanel` |
| Footer | `Footer` |

Deliberate changes: the header is `sticky` instead of `fixed` (the design's fixed header covers
content when the second tab row appears on small screens); all four screens use the universal
screen's 12-column grid (the MoE mockup used a fixed 380 px column); decorative numbers and
labels in the mockups (fake latency, GPU, version strings, SSIM, LPIPS) are replaced with values
from the API or removed.

## Request rules (see docs/api.md)

- Corruption **None** on a sample sends `corruption=clean` (true class known: PSNR and oracle
  routing work). **None** on an upload omits `corruption`: the image is restored as uploaded.
- A corruption type sends `level`, or the custom parameters when "Custom parameters" is on, and
  the seed when one is entered. "Preview corruption" calls `/api/corrupt` and keeps the returned
  seed, so the next restore uses exactly the previewed image.
- Oracle routing is offered only when the server applies the corruption.
- PSNR and SSIM (from `metrics`, against the clean original) are shown under the Model input and
  Restored output panels, as in the Stitch design; they exist only when the server applied the corruption.
- Errors are shown with the backend's `detail` text (400, 404, 413, 415, 422, 500, 503); files
  larger than the limit from `/api/options` are rejected before uploading.
