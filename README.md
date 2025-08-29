# Architecture Shazam
> This project turns music into architectural visuals.
It ingests an audio snippet, extracts MIR features, uses an LLM to write a structured architect’s brief, then renders images (SDXL) and 3D. The research focus is repeatability: do we get coherent series across re-runs, and what reduces variance more - prompt freezing or more diffusion steps?


## Key Features
- End-to-end pipeline: Audio -> MIR features/metadata -> LLM brief -> SDXL image / 3D.
- Prompt mediation: Two-stage LLM (“verbose brief” -> compact SDXL prompt), with optional prompt freezing.
- Repeatability metrics: aHash/dHash/pHash + palette cosine -> Combined Repeatability Score (0–100).
- Reproducibility: Per-output sidecar JSON + structured logs capture exact prompts, token counts, model/scheduler/steps, and environment.
- Web UI: React SPA for uploading/searching previews, generating images/3D, and managing outputs.
- Portable backend: Flask API + SQLAlchemy (SQLite in dev, PostgreSQL in prod).


## Architecture (bird's-eye)
```
React SPA  ->  Flask API  ->  Services
                ├── Analysis (librosa, taggers)
                ├── LLM brief + prompt distillation
                ├── Image (SDXL via diffusers)
                └── 3D (Meshy service or Shap-E fallback)
Persistence: SQLAlchemy (SQLite/PostgreSQL)
Observability: structured logs + per-output sidecar JSON
```


## Reasearch Questions
- RQ1 / H1: For a fixed (song, building type, LLM), do repeated runs produce similar visuals?
    - Findings: Yes — moderate repeatability, strongest in palette and massing.
- RQ2 / H2: What contributes more variance - prompting or the renderer?
    - Finding: Prompt freezing yields larger and more reliable gains than merely increasing denoising steps (with diminishing returns past mid-range steps).

See the report’s Results chapter for details, tables, and exemplars.


## Quickstart
### Requirements
- Python 3.10+ (3.11 recommended)
- Node 18+ (for the SPA)
- A CUDA-capable GPU is recommended for SDXL; GPU(MPS on silicon Mac)/CPU works but is slow.
- API keys for:
    - ACRCloud (fingerprinting)
    - Meshy (3D)
    - Spotify

### Setup
#### 1) Clone & set up backend

```bash
git clone https://git.cs.bham.ac.uk/projects-2024-25/mxm1646.git

cd mxm1646/

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 2) Create .env file
```
# ./.env

# ACRCloud
ACRCLOUD_HOST=
ACRCLOUD_ACCESS_KEY=
ACRCLOUD_ACCESS_SECRET=

# Spotify
SPOTIPY_CLIENT_ID=
SPOTIPY_CLIENT_SECRET=

# Meshy
MESHY_API_KEY=
```

#### 3) Set up frontend
```bash
cd ./client

npm install # first time

npm run dev
npm run api # run in a different terminal to run flask
```
Open the printed local URL (set to http://localhost:8000/ but could be different depending on .flaskenv setup)


## Configuration & Controls
- Fixed (by default):  
Scheduler = DPM-Solver++ · Guidance = 6.5 · Resolution = 1024×1024 · No negative prompt · Batch size = 1
- Manipulated:  
inference steps ∈ {10, 25, 50} · prompt freezing

- Tokenisation:  
Compact SDXL prompts are clamped to ≤77 tokens on both CLIP encoders to avoid silent truncation (repo IDs and counts stored in sidecars).


## Running the Experiments
*The report includes the full protocol and results; below is a practical outline.*

### Calibration (noise-only)
- Freeze the prompt; sweep steps {10, 25, 50}.
- Compute a simple composite noise index (Laplacian var, HF ratio, residual RMS).
- Rule: pick the smallest step count within a small tolerance of the minimum median noise (we used 25).

### H1 (Within-song repeatability)
- Cross songs × building types × LLMs; 10 trials per cell at steps=25.
- Score with aHash/dHash/pHash + palette cosine; combine to 0–100.

### H2 (Prompt vs. renderer)
- Fix LLM (deepseek-r1:14b) and building type (house).
- Compare frozen vs unfrozen prompts at steps {10, 25, 50}.
- Report per-song deltas and medians; inspect component trends (palette vs hashes).

## Metrics (at a glance)
- **aHash:** coarse luminance layout
- **dHash:** local gradients/edges
- **pHash:** DCT global structure (robust to minor shifts)
- **Palette cosine:** Score (0-100) is a weighted average of the four similarities.

*Weights can be adjusted; when a metric is disabled, weights renormalise.*

## LSEPI (Legal, Social, Ethical, Professional)
- Legal/ToS: Use short public previews for catalogue audio; do not store raw audio. Respect ACRCloud/Meshy terms and model licences.
- Privacy: Sidecars/logs omit raw audio; store only features, prompts, and outputs. Provide a data-deletion path.
- Bias/Transparency: Classifier/KB biases can propagate; we log tags sent to the LLM and clearly label AI-generated outputs.
- Reproducibility: Pinned dependencies and environment capture; single-host per condition.

*This is informational, not legal advice.*


## Tech Stack
- Frontend: React (SPA)
- Backend: Flask, SQLAlchemy (SQLite/PostgreSQL)
- Audio/MIR: librosa, ACRCloud, MusicBrainz/Wikipedia for metadata
- LLM: open-weights via local runner (e.g., ollama) or API; two-stage prompting
- Image: SDXL via Hugging Face diffusers
- 3D: Meshy service (primary) -> Shap-E fallback (local)
- Analysis: Python + notebooks/scripts; CSV outputs

## Acknowledgements
Hugging Face diffusers/transformers, librosa, MusicBrainz/Wikipedia, ACRCloud, Meshy, Shap-E, and SDXL-family research cited in the report.


## Contributions
**Closed** for now; may open later. Please avoid committing raw audio or secrets.


## License
This project is licensed under the MIT License.

**Notes:** Third-party models, weights, and services (e.g., SDXL, Meshy, ACRCloud) have their own licenses/ToS; this repo’s MIT license does not grant rights to redistribute those assets.