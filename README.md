# Platonic Ideal

A tool for extracting and analyzing token embeddings from language models to compute semantic "essence" vectors from word groups.

## Purpose

This project extracts **non-contextual token embeddings** from language models (currently tested with Qwen/Qwen2.5-0.5B) and provides a mathematically rigorous method to:

1. **Extract token vectors** - the raw, static representations from the model's embedding matrix
2. **Compute word-group essence vectors** - mathematical aggregation of word meanings with configurable weights
3. **Find nearest dictionary words** - discover which real words are most semantically aligned with your word-group essence

Unlike contextual embeddings (which require a full forward pass through the model), token vectors are the **"platonic ideal"** of how each token is represented before any contextual processing.

## Mathematical Framework

### Token Embeddings

For a language model with vocabulary size \(V\) and embedding dimension \(D\), the embedding matrix is:

\[
\mathbf{E} \in \mathbb{R}^{V \times D}
\]

where each row \(\mathbf{E}[i] \in \mathbb{R}^D\) is the token vector for token ID \(i\).

### Entry Vector (Word/Phrase Embedding)

For an entry (word or phrase) \(e\) that tokenizes into \(T_e\) tokens with IDs \([t_1, t_2, \ldots, t_{T_e}]\), the **entry vector** is the mean of its token embeddings:

\[
\mathbf{v}_e = \frac{1}{T_e} \sum_{j=1}^{T_e} \mathbf{E}[t_j]
\]

This handles multi-token entries (e.g., "solar system" → 2 tokens) by averaging their embeddings.

### Group Mean Vector

For a word group \(g\) containing \(N_g\) entries \(\{e_1, e_2, \ldots, e_{N_g}\}\), the **group mean vector** is:

\[
\mathbf{m}_g = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbf{v}_{e_i}
\]

This ensures each entry contributes equally within its group, regardless of how many tokens it contains.

### Weighted Group Vector

Each group \(g\) has an associated weight \(w_g \in \mathbb{R}\). The **weighted group vector** is:

\[
\mathbf{w}_g = w_g \cdot \mathbf{m}_g
\]

Notes:
- **Positive weights** (\(w_g > 0\)) pull the essence vector toward that group's semantic space
- **Negative weights** (\(w_g < 0\)) push the essence vector away from that group
- **Zero weight** (\(w_g = 0\)) excludes the group entirely

### Overall Essence Vector

Given \(G\) word groups, the **overall essence vector** is the mean of all weighted group vectors:

\[
\mathbf{V}_{\text{essence}} = \frac{1}{G} \sum_{g=1}^{G} \mathbf{w}_g
\]

This ensures **each group contributes equally** to the final essence vector, regardless of how many entries each group contains.

**Key property**: A group with 100 entries has the same influence as a group with 1 entry, as long as their weights are equal.

### Nearest Dictionary Words

To find dictionary words most aligned with the essence vector, we:

1. **Build candidate set**: Extract all lemmas from WordNet with specified part-of-speech (default: nouns and verbs)
2. **Compute word vectors**: For each candidate word \(w\), compute its vector using the same entry-vector method:
   \[
   \mathbf{v}_w = \frac{1}{T_w} \sum_{j=1}^{T_w} \mathbf{E}[t_j]
   \]
   where \(w\) tokenizes into \(T_w\) tokens.

3. **Cosine similarity**: Rank candidates by cosine similarity to the essence vector:
   \[
   \text{sim}(w) = \frac{\mathbf{v}_w \cdot \mathbf{V}_{\text{essence}}}{\|\mathbf{v}_w\| \cdot \|\mathbf{V}_{\text{essence}}\|}
   \]

4. **Return top-k**: Output the \(k\) dictionary words with highest similarity scores

### Why Cosine Similarity?

Cosine similarity measures **directional alignment** in embedding space, not Euclidean distance. Two vectors are similar if they point in the same direction, regardless of magnitude:

\[
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} \in [-1, 1]
\]

where:
- \(\cos(\theta) = 1\): vectors point in the same direction (maximally similar)
- \(\cos(\theta) = 0\): vectors are orthogonal (unrelated)
- \(\cos(\theta) = -1\): vectors point in opposite directions (maximally dissimilar)

This is appropriate for embeddings because:
- Token embeddings have varying norms based on frequency/importance
- We care about semantic direction, not magnitude
- It's the standard metric for word similarity in NLP

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. Download WordNet data (one-time):
```bash
python -m nltk.downloader wordnet omw-1.4
```

## Usage

### Basic Token Extraction

Extract embeddings for specific token IDs:

```bash
python extract_embeddings.py --token-id 100
```

Extract embeddings for text:

```bash
python extract_embeddings.py --text "Hello world"
```

### Word-Group Essence Vector (Main Method)

Create a JSON file with your word groups (e.g., `groups.json`):

```json
{
  "repo_id": "Qwen/Qwen2.5-0.5B",
  "groups": [
    {
      "name": "space",
      "weight": 1.0,
      "entries": ["planet", "star", "sun", "earth", "galaxy", "solar system", "orbit", "astronomy"]
    },
    {
      "name": "ocean",
      "weight": 1.0,
      "entries": ["ocean", "sea", "wave", "coral reef", "tide", "marine"]
    },
    {
      "name": "technology",
      "weight": 0.5,
      "entries": ["computer", "software", "algorithm", "programming"]
    }
  ]
}
```

Run the essence computation:

```bash
python word_group_essence_wordnet.py --input groups.json --top-k 20 --output groups.out.json
```

Output JSON structure:
```json
{
  "repo_id": "Qwen/Qwen2.5-0.5B",
  "groups": [ ... ],
  "_output": {
    "model": { "repo_id": "...", "vocab_size": 151936, "embedding_dim": 896 },
    "settings": { "wordnet_pos": ["n", "v"], "exclude_substrings": true, ... },
    "groups": [
      {
        "index": 0,
        "name": "space",
        "weight_input": 1.0,
        "weight_used": 1.0,
        "num_entries": 8,
        "entries": [ { "text": "planet", "token_ids": [...], "tokens": [...] }, ... ],
        "group_mean_norm": 0.243,
        "group_weighted_norm": 0.243
      },
      ...
    ],
    "overall": { "norm": 0.195 },
    "top_words": [
      { "word": "moon", "cosine_similarity": 0.497, "token_ids": [...] },
      { "word": "astronaut", "cosine_similarity": 0.490, "token_ids": [...] },
      ...
    ],
    "warnings": []
  }
}
```

### Command-Line Options

#### Core Options
- `--input FILE` - Path to input JSON file (required)
- `--output FILE` - Path to write output JSON (if omitted, prints to stdout)
- `--repo-id REPO_ID` - Override model repository (default: from JSON or `Qwen/Qwen2.5-0.5B`)
- `--top-k K` - Number of nearest words to return (default: 20)

#### Dictionary Options
- `--wordnet-pos POS` - Comma-separated list of WordNet POS tags to include (default: `n,v` for nouns and verbs)
  - `n` = nouns
  - `v` = verbs
  - `a` = adjectives
  - `r` = adverbs
  - Example: `--wordnet-pos n` for nouns only

#### Filtering Options
- `--exclude-input / --no-exclude-input` - Exclude input words from results (default: on)
- `--exclude-substrings / --no-exclude-substrings` - Exclude candidates containing input words as substrings (default: on)
  - Example: input `earth` won't return `earthworm`, `earthquake`, etc.
- `--min-word-chars N` - Minimum character length for candidate words (default: 3)
- `--max-token-len N` - Skip words that tokenize into more than N tokens (default: 6)

#### Performance Options
- `--batch-size N` - Batch size for embedding loading (default: 8192)
- `--candidate-batch N` - Batch size for scoring candidate words (default: 4096)
- `--weight-clip-abs X` - Clip |group weight| to this value (default: 32.0)
- `--no-weight-clip` - Disable weight clipping

## Example Workflow

### Example 1: Computing Essence for Space-Related Words

Input (`space.json`):
```json
{
  "groups": [
    { "weight": 1.0, "entries": ["planet", "star", "sun", "galaxy", "orbit", "astronomy"] }
  ]
}
```

Run:
```bash
python word_group_essence_wordnet.py --input space.json --top-k 10
```

Expected top words: `moon`, `astronaut`, `telescope`, `cosmic`, etc.

### Example 2: Combining Multiple Concepts with Different Weights

Input (`mixed.json`):
```json
{
  "groups": [
    { "name": "nature", "weight": 1.0, "entries": ["tree", "forest", "river", "mountain"] },
    { "name": "urban", "weight": -0.5, "entries": ["city", "building", "street", "urban"] }
  ]
}
```

This creates an essence vector that is:
- **Attracted to** nature words (weight = 1.0)
- **Repelled from** urban words (weight = -0.5)

Expected top words: wilderness/nature-related terms that are semantically distant from cities.

### Example 3: Finding Words Orthogonal to a Concept

Input (`not_food.json`):
```json
{
  "groups": [
    { "weight": -1.0, "entries": ["food", "eat", "meal", "hungry", "restaurant"] }
  ]
}
```

With negative weight, the essence vector points **away from** food-related concepts.

Expected top words: non-food words (technology, abstract concepts, etc.)

## Technical Details

### Model Architecture

For Qwen/Qwen2.5-0.5B:
- **Vocabulary size**: 151,936 tokens
- **Embedding dimension**: 896
- **Data type**: bfloat16 (converted to float32 for numpy compatibility)
- **File format**: single `model.safetensors` file (~1GB)

The embedding matrix is located at tensor name `model.embed_tokens.weight` within the safetensors file.

### Tokenization Details

The tokenizer uses **subword encoding**, which means:
- Some words are single tokens: `"planet"` → `[50074]`
- Some words are multiple tokens: `"solar system"` → `[42578, 1080]`
- Some words split unexpectedly: `"astronomy"` → `[20467, 16974]` = `["astr", "onomy"]`

The entry vector computation (mean of token vectors) ensures that multi-token entries are handled correctly.

### Dictionary Source (WordNet)

**WordNet** is a large lexical database of English developed by Princeton University. It contains:
- **Nouns**: ~117,000 lemmas (e.g., `planet`, `ocean`, `computer`)
- **Verbs**: ~11,000 lemmas (e.g., `swim`, `orbit`, `compute`)
- **Adjectives**: ~22,000 lemmas
- **Adverbs**: ~4,500 lemmas

By default, we use **nouns and verbs** (~128k candidates) because they typically represent concrete concepts better than adjectives/adverbs.

Each WordNet entry represents a **lemma** (base form), so results are canonical:
- `planets` → included as `planet`
- `swimming` → included as `swim`
- `computed` → included as `compute`

### Why Dictionary Filtering Matters

Without dictionary filtering, the nearest neighbors in embedding space are often:
- **Subword tokens**: `n`, `m`, `ch`, `ig` (single letters or bigrams)
- **Rare compounds**: `earthporn`, `nch` (corpus artifacts, not real words)
- **Fragments**: pieces of words rather than complete words

Dictionary filtering (via WordNet) ensures results are **actual English words** that appear in a linguistic database, not just statistically frequent token sequences.

### Performance Characteristics

For the word-group essence computation with WordNet dictionary (~60k candidates after filtering):

- **Loading embeddings**: ~5-10 seconds (one-time per session)
- **Computing essence vector**: <1 second for typical inputs (5-20 entries)
- **Scoring candidates**: ~10-30 seconds for 60k words
  - Batched computation (4096 words/batch)
  - Vectorized cosine similarity
  - Min-heap for top-k selection

Memory usage:
- **Embedding matrix**: ~550 MB (151k × 896 × 4 bytes)
- **Candidate word vectors**: ~220 MB peak during scoring (60k × 896 × 4 bytes)
- **Total peak**: ~800 MB (CPU-only, no GPU needed)

## Installation

### Backend Setup

1. **Create Virtual Environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
pip install -r backend/requirements-backend.txt
```

Key dependencies:
- `transformers>=4.40.0` + `tokenizers>=0.19.1` - Model tokenizers
- `torch>=2.2.0` - Handle bfloat16 tensors (CPU-only)
- `numpy<2.0.0` - Vector operations (pinned for compatibility)
- `fastapi` + `uvicorn` - Web API framework
- `nltk>=3.9.1` + `wordfreq>=3.1.1` - Dictionary data

3. **Download WordNet Data** (one-time):
```bash
python -m nltk.downloader wordnet omw-1.4
```

### Frontend Setup

1. **Install Node.js Dependencies**:
```bash
cd frontend
npm install
```

Key dependencies:
- `next@16.1.0` + `react@19` - Framework
- `@dnd-kit/*` - Drag-and-drop groups
- `@radix-ui/*` - UI components (shadcn/ui)
- `tailwindcss@4` - Styling

## Configuration (Environment Variables)

### Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_MODELS` | `tinyllama-1.1b,qwen-0.5b` | Comma-separated list of allowed model IDs |
| `PORT` | `8000` | Server port (Railway uses dynamic `$PORT`) |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allowed origins |
| `REDIS_URL` | (none) | Redis connection string for distributed session locking |
| `PLATONIC_SESSION_LOCK_TTL_SECONDS` | `1200` | Session lock timeout (20 minutes) |

### Frontend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | `http://127.0.0.1:8000` | Backend API URL (must be set for production) |

## Quick Start

### Option 1: Run the Web App (Recommended)

**Backend**:
```bash
source venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**Frontend** (in another terminal):
```bash
cd frontend
npm run dev
```

Visit: `http://localhost:3000`

### Option 2: CLI Usage

#### Extract Token Embeddings

Get the embedding vector for a specific token ID:
```bash
python extract_embeddings.py --token-id 100
```

Get embeddings for text (shows tokenization + vectors):
```bash
python extract_embeddings.py --text "Hello world"
```

#### Compute Word-Group Essence

1. Create `groups.json`:
```json
{
  "repo_id": "Qwen/Qwen2.5-0.5B",
  "groups": [
    {
      "name": "space",
      "weight": 1.0,
      "entries": ["planet", "star", "sun", "earth", "galaxy", "orbit", "astronomy"]
    },
    {
      "name": "ocean",
      "weight": 1.0,
      "entries": ["ocean", "sea", "wave", "tide", "marine"]
    }
  ]
}
```

2. Run:
```bash
python word_group_essence_wordnet.py --input groups.json --top-k 20 --output groups.out.json
```

3. View results:
```bash
python - <<'PY'
import json
with open('groups.out.json') as f:
    d = json.load(f)
for i, w in enumerate(d['_output']['top_words'], 1):
    print(f"{i:2d}. {w['word']:20s} similarity={w['cosine_similarity']:.4f}")
PY
```

## Full-Stack Application

This repository includes a **production-ready web application** with:
- **Backend (FastAPI)**: Exposes essence computation as JSON APIs
- **Frontend (Next.js)**: Beautiful, interactive UI for word-group exploration
- **CI/CD Pipeline**: Automated testing via GitHub Actions
- **Deployment**: Ready for Railway (backend) + Vercel (frontend)

### Architecture

```
Frontend (Next.js)          Backend (FastAPI)           Models (On-Demand)
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│ React UI        │──HTTP──│ /health          │        │ TinyLlama 1.1B  │
│ Word Groups     │        │ /compute-essence │───────▶│ Qwen 0.5B       │
│ Model Selection │        │ Session Lock     │        │ (Downloads      │
│ Results Display │        │ CORS             │        │  on first use)  │
└─────────────────┘        └──────────────────┘        └─────────────────┘
```

### Key Features

**Backend**:
- **Model Allowlist**: Configure which models are available via `ALLOWED_MODELS` env var (defaults to `tinyllama-1.1b,qwen-0.5b`)
- **Session Locking**: Prevents concurrent requests per user (uses Redis if available, or in-memory fallback)
- **Dynamic CORS**: Configurable via `ALLOWED_ORIGINS` for production domains
- **On-Demand Loading**: Models load only when requested (no persistent RAM cache)

**Frontend**:
- **Dynamic Model Discovery**: Fetches available models from backend `/health` endpoint
- **Drag-and-Drop Groups**: Reorder word groups with visual feedback
- **Multi-Model Comparison**: Analyze with up to 3 models simultaneously
- **Responsive Design**: Built with Tailwind CSS and shadcn/ui components

### Run Locally

**Backend** (Terminal 1):
```bash
source venv/bin/activate
ALLOWED_MODELS="tinyllama-1.1b,qwen-0.5b" PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**Frontend** (Terminal 2):
```bash
cd frontend
npm install  # First time only
npm run dev
```

Then visit: `http://localhost:3000`

> **Note**: Backend defaults to port 8000, frontend defaults to `http://127.0.0.1:8000` for API calls.

### Deploy to Production

**Backend → Railway**:
1. Connect your GitHub repo to Railway
2. Railway auto-detects the `Dockerfile`
3. Set environment variables:
   ```
   ALLOWED_MODELS=tinyllama-1.1b,qwen-0.5b
   PORT=8000
   ALLOWED_ORIGINS=https://your-frontend.vercel.app
   ```
4. Deploy! (First build takes ~5-10 min to download models)

**Frontend → Vercel**:
1. Connect your GitHub repo to Vercel
2. Set **Root Directory**: `frontend`
3. Add environment variable:
   ```
   NEXT_PUBLIC_API_BASE_URL=https://your-backend.up.railway.app
   ```
4. Deploy!

**Auto-Deploy**: Both platforms automatically deploy on every push to `main`. GitHub Actions runs tests first to catch errors.

### API Endpoints

#### `GET /health`
Returns server status and available models.

**Response**:
```json
{
  "status": "ok",
  "default_model_id": "tinyllama-1.1b",
  "available_models": [
    {
      "id": "tinyllama-1.1b",
      "name": "TinyLlama 1.1B (Recommended)",
      "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "speed": "fast",
      "description": "TinyLlama 1.1B (best balance of quality/speed from tests)"
    }
  ]
}
```

#### `POST /compute-essence`
Computes word-group essence and returns nearest dictionary words.

**Headers**:
- `X-Session-ID`: Required (prevents concurrent requests per session)

**Request Body**:
```json
{
  "model_ids": ["tinyllama-1.1b"],
  "groups": [
    { "name": "space", "weight": 1.0, "entries": ["planet", "star", "sun", "galaxy"] }
  ],
  "options": {
    "top_k": 20,
    "wordnet_pos": "n,v",
    "exclude_input": true,
    "exclude_substrings": true
  }
}
```

### Request/Response (compute-essence)
**Request**
```json
{
  "model_ids": ["tinyllama-1.1b", "qwen-0.5b"],
  "groups": [
    { "name": "space", "weight": 1.0, "entries": ["planet", "star", "sun", "galaxy", "orbit", "astronomy"] }
  ],
  "options": {
    "top_k": 20,
    "wordnet_pos": "n,v",
    "exclude_input": true,
    "exclude_substrings": true,
    "min_word_chars": 3,
    "max_token_len": 6
  }
}
```

**Response** (truncated)
```json
{
  "model_ids": ["tinyllama-1.1b", "qwen-0.5b"],
  "results": {
    "tinyllama-1.1b": { "...": "full per-model result (same shape as CLI output)" },
    "qwen-0.5b": { "...": "full per-model result" }
  },
  "errors": {},
  "timing_s": 2.7,
  "per_model_timing_s": { "tinyllama-1.1b": 1.8, "qwen-0.5b": 0.9 }
}
```

### Input validation defaults
- Max groups: 10
- Max entries per group: 50
- Max entry length: 100 chars
- `top_k`: 1..100
- WordNet POS: `n,v,a,r,s` (default `n,v`)

### Available Models

The backend supports 7 models (configured in `backend/app/config.py`):

| Model ID | Name | Repo ID | Speed | RAM (Est.) |
|----------|------|---------|-------|------------|
| `tinyllama-1.1b` | TinyLlama 1.1B (Recommended) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | fast | ~250MB |
| `qwen-0.5b` | Qwen 0.5B | `Qwen/Qwen2.5-0.5B` | fast | ~550MB |
| `qwen-1.5b` | Qwen 1.5B | `Qwen/Qwen2.5-1.5B` | medium | ~800MB |
| `qwen-3b` | Qwen 3B | `Qwen/Qwen2.5-3B` | slow | ~1.5GB |
| `qwen-7b` | Qwen 7B (Large) | `Qwen/Qwen2.5-7B` | very slow | ~3GB |
| `phi-2` | Phi-2 2.7B | `microsoft/phi-2` | slow | ~1GB |
| `gemma-2b` | Gemma 2B | `google/gemma-2b` | medium | ~900MB |

**Model Filtering**: Use the `ALLOWED_MODELS` environment variable to control which models are available:
```bash
# Only allow lightweight models (default for production)
export ALLOWED_MODELS="tinyllama-1.1b,qwen-0.5b"

# Allow all models (requires more RAM)
export ALLOWED_MODELS="tinyllama-1.1b,qwen-0.5b,qwen-1.5b,qwen-3b,qwen-7b,phi-2,gemma-2b"
```

The frontend dynamically fetches the allowed models from the backend's `/health` endpoint, so no frontend changes are needed.

> **Note**: Models are loaded on-demand per request. First run downloads the model files from Hugging Face. Subsequent runs use the cached files.

## CI/CD Pipeline

This project includes automated testing via **GitHub Actions** (`.github/workflows/ci.yml`).

### What Gets Tested

Every push to `main` triggers three parallel jobs:

1. **Backend Tests** (Python)
   - Lints code with `flake8`
   - Runs `pytest` tests
   - Downloads NLTK data
   - Validates Python 3.11 compatibility

2. **Frontend Tests** (Next.js)
   - Lints code with ESLint
   - Type-checks with TypeScript
   - Builds production bundle
   - Validates Node.js 20 compatibility

3. **Docker Build** (on PRs only)
   - Validates `Dockerfile` and `frontend/Dockerfile`
   - Ensures deployment readiness

### Viewing CI Results

Visit: `https://github.com/YOUR-USERNAME/platonic-ideal/actions`

- ✅ Green checkmark = All tests passed
- ❌ Red X = Tests failed (click for details)

### Deployment Triggers

- **Railway**: Auto-deploys backend on every push to `main`
- **Vercel**: Auto-deploys frontend on every push to `main`
- **Safeguard**: If CI tests fail, deployment proceeds anyway (Railway/Vercel don't block on GitHub Actions)

For maximum safety, enable "Required Status Checks" in GitHub repository settings to block merges if CI fails.

## Understanding the Results

### What Does the Essence Vector Represent?

The essence vector is a **mathematical aggregation** of the semantic directions represented by your input word groups. It captures:

- The **average semantic space** of your inputs
- The **relative emphasis** determined by group weights
- The **geometric center** (or offset center, if using negative weights)

### Interpreting Top Words

Top words with high cosine similarity are those whose embeddings **point in a similar direction** as your essence vector in the \(\mathbb{R}^{896}\) embedding space.

**Example**: If your groups are `space` and `ocean`:
- Expected top words: **water**, **sky**, **liquid**, **blue** (concepts bridging both)
- If you see **moon**: makes sense (related to space + affects ocean tides)
- If you see **astronomy**: very expected (directly in your input)
- If you see **moonflower**: compound word, less meaningful

### Why You See Compound Words

WordNet includes many **compound nouns** like:
- `moonshine`, `moonflower`, `moonbeam` (all contain "moon")
- `earthquake`, `earthworm`, `earthstar` (all contain "earth")

These compounds rank highly because:
1. They share a token with input words → their vectors are close to the mean
2. They're valid dictionary words in WordNet

**Solution**: Use `--exclude-substrings` (default: on) to exclude candidates containing input words as substrings.

### Adjusting Results Quality

If top words seem off:

1. **Add more entries** to each group to better define the concept
2. **Adjust weights** to emphasize important groups
3. **Use `--wordnet-pos n` for nouns only** (excludes verbs)
4. **Increase `--top-k`** to see more candidates (e.g., 50)
5. **Check your input words** - ensure they're semantically coherent within each group

## Project Structure

```
platonic-ideal/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # API endpoints + CORS
│   │   ├── config.py          # Model registry + allowlist
│   │   ├── models.py          # Pydantic schemas
│   │   ├── compute.py         # Essence computation logic
│   │   └── session_lock.py    # Redis/in-memory session locking
│   ├── tests/
│   │   └── test_api.py        # Pytest tests
│   └── requirements-backend.txt
│
├── frontend/                   # Next.js frontend
│   ├── app/
│   │   ├── page.tsx           # Main UI component
│   │   ├── layout.tsx         # App layout
│   │   └── globals.css        # Global styles
│   ├── components/ui/         # shadcn/ui components
│   ├── package.json
│   └── Dockerfile             # Frontend container
│
├── extract_embeddings.py      # CLI: Extract token embeddings
├── word_group_essence_wordnet.py  # CLI: Compute essence (WordNet)
├── test_models.py             # CLI: Compare multiple models
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Backend container
├── docker-compose.yml         # Local multi-service setup
└── .github/workflows/ci.yml   # CI/CD pipeline
```

### Key Files

**Core Scripts**:
- `extract_embeddings.py` - Download and load embedding matrices
- `word_group_essence_wordnet.py` - Compute essence + find nearest words
- `test_models.py` - Compare results across models

**Backend**:
- `backend/app/main.py` - API server with CORS and session locking
- `backend/app/config.py` - Model registry and `ALLOWED_MODELS` filtering
- `backend/app/compute.py` - Async essence computation wrapper

**Frontend**:
- `frontend/app/page.tsx` - React UI with drag-and-drop groups
- `frontend/components/ui/` - Reusable UI components (buttons, cards, sliders)

**Deployment**:
- `Dockerfile` - Backend container (Railway)
- `frontend/Dockerfile` - Frontend container (multi-stage build)
- `docker-compose.yml` - Run both services locally with Docker

## Limitations

### Model-Specific
- **Tested only on Qwen/Qwen2.5-0.5B** - other models may use different tensor names or formats
- **English-centric** - WordNet is primarily for English; extend with multilingual lexicons for other languages
- **Subword tokenization** - some semantically meaningful units may not be single tokens

### Mathematical
- **Linear averaging** - the mean operation assumes semantic dimensions are additive; may not capture all relationships
- **Static embeddings** - these are non-contextual; a word like "bank" has one vector for both "river bank" and "financial bank"
- **Cosine similarity** - measures directional alignment only; magnitude information is discarded

### Practical
- **Computational**: scoring 60k+ WordNet candidates takes 10-30 seconds on CPU
- **Memory**: requires ~800 MB RAM for full operation
- **Dictionary coverage**: WordNet doesn't include all proper nouns, neologisms, slang, or domain-specific terminology

## Future Enhancements

Potential improvements:
- Support for more language models (Llama, GPT-2, BERT variants)
- Multilingual dictionary support (WordNet equivalents for other languages)
- FAISS-based approximate nearest neighbor search for faster candidate scoring
- Weighted word-group combinations with learned optimal weights
- Integration with contextualized embeddings for comparison
- Package structure for `pip install platonic-ideal`

## Citation & Acknowledgments

This project uses:
- **Hugging Face Transformers** for tokenizer and model utilities
- **safetensors** for efficient weight loading
- **NLTK WordNet** for English dictionary lemmas
- **Qwen/Qwen2.5-0.5B** model from Alibaba Cloud

## License

(Add your license here)

