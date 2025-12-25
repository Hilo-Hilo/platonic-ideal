# Platonic Ideal

A tool for extracting and analyzing token embeddings from language models to compute semantic "essence" vectors from word groups.

## Purpose

This project extracts **non-contextual token embeddings** from language models (currently tested with Qwen/Qwen2.5-0.5B) and provides a mathematically rigorous method to:

1. **Extract token vectors** - the raw, static representations from the model's embedding matrix
2. **Compute word-group essence vectors** - mathematical aggregation of word meanings with configurable weights
3. **Find nearest dictionary words** - discover which real words are most semantically aligned with your word-group essence

Unlike contextual embeddings (which require a full forward pass through the model), token vectors are the **"platonic ideal"** of how each token is represented before any contextual processing.

## Mathematical Framework

### V2 Improvements (New!)

The v2 endpoint implements three mathematically-grounded improvements for better semantic retrieval:

#### 1. Spherical Averaging

**Problem**: Token embeddings have varying norms (often correlated with frequency). Averaging unnormalized vectors lets high-norm tokens dominate.

**Solution**: L2-normalize each token embedding before averaging, then normalize the result:

\[
\hat{\mathbf{E}}[t] = \frac{\mathbf{E}[t]}{\|\mathbf{E}[t]\|_2}, \quad
\mathbf{v}_e = \text{normalize}\left(\frac{1}{T_e} \sum_{j=1}^{T_e} \hat{\mathbf{E}}[t_j]\right)
\]

This averages **directions** on the unit sphere, matching cosine-based retrieval better than Euclidean averaging.

#### 2. Per-Group Scoring

**Problem**: Collapsing positive and negative weighted groups into a single essence vector can cause cancellation, producing weak or generic directions.

**Solution**: Score each candidate word against each group separately:

\[
\text{score}(w) = \sum_{g=1}^{G} w_g \cdot \cos(\mathbf{v}_w, \mathbf{m}_g)
\]

This makes negative weights work as literal **repulsion** rather than vector subtraction.

#### 3. All-but-the-Top (Anisotropy Correction)

**Problem**: Static embeddings are anisotropic—many vectors share common components that inflate cosine similarity in uninteresting directions.

**Solution**: Remove mean and top \(k\) principal components:

\[
\mathbf{v}' = (\mathbf{I} - \mathbf{U}_k \mathbf{U}_k^\top)(\mathbf{v} - \boldsymbol{\mu})
\]

where \(\boldsymbol{\mu}\) is the mean of candidate vectors and \(\mathbf{U}_k\) contains the top \(k\) principal directions.

This significantly improves semantic similarity tasks (see [arXiv:1702.01417](https://arxiv.org/abs/1702.01417)).

#### 4. Tokenization Invariance (v2.1)

**Problem**: Tokenizers split words differently depending on whitespace. `"apple"` vs `" apple"` may produce different token sequences, leading to inconsistent embeddings.

**Solution**: Average embeddings with and without leading space:

\[
\mathbf{v}_{\text{word}} = \text{normalize}\left(\frac{1}{2}\left(\mathbf{v}(\text{word}) + \mathbf{v}(\text{" word"})\right)\right)
\]

This reduces position-dependent tokenization artifacts while staying non-contextual.

#### 5. Word Frequency Filtering (v2.1)

**Problem**: WordNet contains rare, archaic, and technical terms that rarely appear in practice.

**Solution**: Filter candidates to common English words using `wordfreq`:

\[
\text{candidates} = \text{WordNet} \cap \text{wordfreq.top\_n\_list}(\text{"en"}, N=200000)
\]

This ensures results are recognizable, common words rather than obscure lemmas.

#### 6. Robust Group Centers (v2.2)

**Problem**: Outlier entries (polysemous words, typos, semantic drift) can skew the group mean.

**Solution**: Use **trimmed mean** — drop bottom fraction by cosine-to-preliminary-mean:

\[
\mathbf{m}_g^{\text{robust}} = \text{mean}\left(\{\mathbf{v}_i : \cos(\mathbf{v}_i, \mathbf{m}_g^{\text{prelim}}) > \text{threshold}\}\right)
\]

Default: drop bottom 20% of entries before computing final group mean.

#### 7. Sense-Aware Reranking (v2.2)

**Problem**: Polysemous words (e.g., "bank", "seal", "bat") have blended embeddings that mix multiple senses.

**Solution**: Rerank using **synset vectors** that mix lemma and gloss (definition):

\[
\mathbf{u}_{\text{synset}} = \alpha \cdot \mathbf{v}_{\text{lemma}} + (1-\alpha) \cdot \mathbf{v}_{\text{gloss}}
\]

where \(\alpha = 0.7\) by default. Score each synset, return the lemma from the best-matching synset.

#### 8. Diagonal Whitening (v2.2)

**Problem**: After All-but-the-Top, some dimensions may still have higher variance than others.

**Solution**: Scale each dimension by inverse standard deviation:

\[
\mathbf{v}'_d = \frac{\mathbf{v}_d - \mu_d}{\sigma_d + \epsilon}
\]

This is a lightweight \(O(N \cdot D)\) whitening that improves isotropy without full covariance computation.

---

### Complete V2.2 Pipeline

The full v2.2 computation pipeline:

1. **Load embeddings**: \(\mathbf{E} \in \mathbb{R}^{V \times D}\)

2. **Compute entry vectors** (tokenization-invariant + spherical):
   \[
   \mathbf{v}_e = \text{normalize}\left(\frac{1}{2}\left(\mathbf{v}_{\text{sph}}(e) + \mathbf{v}_{\text{sph}}(\text{" "} + e)\right)\right)
   \]

3. **Compute group centers** (robust trimmed mean):
   \[
   \mathbf{m}_g = \text{trimmed\_mean}(\{\mathbf{v}_{e_i}\}_{i=1}^{N_g}, p=0.2)
   \]

4. **Build candidate set**: WordNet \(\cap\) wordfreq top 200k

5. **Compute candidate vectors** (same tokenization-invariant + spherical method)

6. **Apply All-but-the-Top** to all vectors (candidates + groups):
   \[
   \mathbf{v}' = (\mathbf{I} - \mathbf{U}_5 \mathbf{U}_5^\top)(\mathbf{v} - \boldsymbol{\mu})
   \]

7. **Apply diagonal whitening**:
   \[
   \mathbf{v}'' = \frac{\mathbf{v}' - \boldsymbol{\mu}'}{\boldsymbol{\sigma} + \epsilon}
   \]

8. **Score candidates** (per-group scoring):
   \[
   \text{score}(w) = \sum_{g=1}^{G} w_g \cdot \cos(\mathbf{v}''_w, \mathbf{m}''_g)
   \]

9. **Sense-aware reranking**: For top candidates, rescore using:
   \[
   \mathbf{u}_s = \text{normalize}(\alpha \cdot \mathbf{v}_{\text{lemma}} + (1-\alpha) \cdot \mathbf{v}_{\text{gloss}})
   \]
   for each synset \(s\), pick best synset score.

10. **Return top-k** words with highest scores

This pipeline combines all 8 improvements for maximum semantic quality.

---

### Token Embeddings (V1 Baseline)

For a language model with vocabulary size \(V\) and embedding dimension \(D\), the embedding matrix is:

\[
\mathbf{E} \in \mathbb{R}^{V \times D}
\]

where each row \(\mathbf{E}[i] \in \mathbb{R}^D\) is the token vector for token ID \(i\).

### Entry Vector (Word/Phrase Embedding)

**V1 (Euclidean mean)**:

For an entry (word or phrase) \(e\) that tokenizes into \(T_e\) tokens with IDs \([t_1, t_2, \ldots, t_{T_e}]\), the **entry vector** is the mean of its token embeddings:

\[
\mathbf{v}_e = \frac{1}{T_e} \sum_{j=1}^{T_e} \mathbf{E}[t_j]
\]

**V2+ (Spherical mean)**:

\[
\mathbf{v}_e = \text{normalize}\left(\frac{1}{T_e} \sum_{j=1}^{T_e} \text{normalize}(\mathbf{E}[t_j])\right)
\]

**V2.1+ (Tokenization-invariant)**:

\[
\mathbf{v}_e = \text{normalize}\left(\frac{1}{2}\left(\mathbf{v}_{\text{spherical}}(e) + \mathbf{v}_{\text{spherical}}(\text{" "} + e)\right)\right)
\]

This handles multi-token entries and reduces tokenization artifacts.

### Group Mean Vector

**V1/V2/V2.1 (Simple mean)**:

For a word group \(g\) containing \(N_g\) entries \(\{e_1, e_2, \ldots, e_{N_g}\}\), the **group mean vector** is:

\[
\mathbf{m}_g = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbf{v}_{e_i}
\]

**V2.2 (Robust trimmed mean)**:

\[
\mathbf{m}_g^{\text{robust}} = \frac{1}{|\mathcal{K}|} \sum_{i \in \mathcal{K}} \mathbf{v}_{e_i}
\]

where \(\mathcal{K}\) contains the top \((1-p)\) fraction of entries by cosine similarity to the preliminary mean (default \(p=0.2\)).

This ensures each entry contributes equally within its group, and outliers are down-weighted in v2.2.

### Weighted Group Vector

Each group \(g\) has an associated weight \(w_g \in \mathbb{R}\). The **weighted group vector** is:

\[
\mathbf{w}_g = w_g \cdot \mathbf{m}_g
\]

Notes:
- **Positive weights** (\(w_g > 0\)) pull the essence vector toward that group's semantic space
- **Negative weights** (\(w_g < 0\)) push the essence vector away from that group
- **Zero weight** (\(w_g = 0\)) excludes the group entirely

### Overall Essence Vector & Scoring

**V1 (Single essence vector)**:

Given \(G\) word groups, the **overall essence vector** is the mean of all weighted group vectors:

\[
\mathbf{V}_{\text{essence}} = \frac{1}{G} \sum_{g=1}^{G} w_g \cdot \mathbf{m}_g
\]

Then score candidates: \(\text{score}(w) = \cos(\mathbf{v}_w, \mathbf{V}_{\text{essence}})\)

**V2+ (Per-group scoring)**:

\[
\text{score}(w) = \sum_{g=1}^{G} w_g \cdot \cos(\mathbf{v}_w, \mathbf{m}_g)
\]

This prevents cancellation between positive and negative weighted groups.

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
- **Backend (FastAPI)**: Exposes essence computation as JSON APIs (v2.2 by default for best quality)
- **Frontend (Next.js)**: Beautiful, interactive UI for word-group exploration
- **CI/CD Pipeline**: Automated testing via GitHub Actions
- **Deployment**: Ready for Railway (backend) + Vercel (frontend)

**Production Version**: The frontend now uses `/compute-essence-v2-2` by default, delivering dramatically better semantic quality with all mathematical improvements active.

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

**Production Configuration**: The application now uses **v2.2 by default**, which delivers 2-3x better semantic quality than v1. All API versions remain available for backwards compatibility.

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
5. **Note**: All new files (`word_group_essence_wordnet_v2*.py`) are automatically included in the Docker build

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

#### `POST /compute-essence-v2` (Mathematical Improvements)
**New in v2**: Enhanced endpoint with three mathematical improvements:

1. **Spherical Averaging**: Normalize token embeddings before averaging (averages directions on unit sphere)
2. **Per-Group Scoring**: Score candidates against each group separately to prevent positive/negative cancellation
3. **All-but-the-Top**: Remove mean + top principal components to correct embedding space anisotropy

**V2-Specific Options**:
- `use_spherical_mean` (default: `true`)
- `use_per_group_scoring` (default: `true`)
- `abt_enabled` (default: `true`)
- `abt_num_components` (default: `5`, range: 1-20)

#### `POST /compute-essence-v2-1` (Quality Filtering)
**New in v2.1**: Adds two highest-ROI search-quality improvements on top of v2:

1. **Word Frequency Filtering**: Filters WordNet candidates to common English words using `wordfreq` (top 200k by default)
2. **Tokenization Invariance**: Averages embeddings with/without leading space to reduce tokenization artifacts

**V2.1-Specific Options**:
- `freq_filter_enabled` (default: `true`)
- `freq_top_n` (default: `200000`, range: 1000-500000)
- `freq_min_zipf` (optional alternative to `freq_top_n`)
- `tokenization_invariance` (default: `true`)

**Request Body Example**:
```json
{
  "model_ids": ["tinyllama-1.1b"],
  "groups": [
    { "name": "space", "weight": 1.0, "entries": ["planet", "star", "sun", "galaxy"] },
    { "name": "ocean", "weight": 1.0, "entries": ["ocean", "sea", "wave", "tide"] }
  ],
  "options": {
    "top_k": 20,
    "freq_filter_enabled": true,
    "freq_top_n": 200000,
    "tokenization_invariance": true
  }
}
```

#### `POST /compute-essence-v2-2` (Advanced Quality - RECOMMENDED)
**New in v2.2**: Adds three advanced quality improvements on top of v2.1:

1. **Robust Group Centers**: Uses trimmed mean (drops outliers by cosine-to-mean) instead of simple averaging
2. **Sense-Aware Reranking**: Reranks candidates using WordNet synset glosses to disambiguate polysemy
3. **Diagonal Whitening**: Lightweight isotropy correction (scales each dimension by inverse std dev)

**V2.2-Specific Options**:
- `robust_group_center` (default: `true`)
- `trim_fraction` (default: `0.2`, range: 0.0-0.5)
- `sense_rerank_enabled` (default: `true`)
- `sense_alpha` (default: `0.7`, range: 0.0-1.0) - mix ratio for lemma vs gloss
- `sense_rerank_pool_multiplier` (default: `10`, range: 1-50)
- `diag_whiten_enabled` (default: `true`)
- `diag_whiten_eps` (default: `1e-6`)

**Performance Comparison** (Space + Ocean test):

| Version | Top Word | Score | Quality | Speed |
|---------|----------|-------|---------|-------|
| V1 | `galax` (fragment) | 0.33 | ⭐⭐ | ~8s |
| V2 | `galax` (fragment) | 0.46 | ⭐⭐⭐ | ~8s |
| V2.1 | `galax` (fragment) | 0.47 | ⭐⭐⭐ | ~8s |
| **V2.2** | **`world`** | **0.91** | **⭐⭐⭐⭐⭐** | ~9s |

**When to use each version**:

| Use Case | Recommended Version |
|----------|---------------------|
| **Production / Best Quality** | **V2.2** (sense-aware + robust + whitening) |
| **Negative weights / Repulsion** | V2+ (per-group scoring) |
| **Common words only** | V2.1+ (wordfreq filtering) |
| **Backwards compatibility** | V1 (original) |
| **Fastest** | V1 (no PCA/reranking) |

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

## Mathematical Improvements (v2, v2.1, v2.2)

### Version Progression

| Version | Key Features | Semantic Quality | Use Case |
|---------|-------------|------------------|----------|
| **V1** | Original (Euclidean averaging, single essence vector) | ⭐⭐ | Backwards compatibility |
| **V2** | Spherical averaging + Per-group scoring + All-but-the-Top | ⭐⭐⭐ | Better math, handles negatives |
| **V2.1** | V2 + Wordfreq filtering + Tokenization invariance | ⭐⭐⭐ | Common words, fewer artifacts |
| **V2.2** | V2.1 + Robust centers + Sense reranking + Diagonal whitening | ⭐⭐⭐⭐⭐ | **RECOMMENDED** |

### V2.2 Delivers Dramatic Improvements

**Test: Space + Ocean Groups**

| Version | Top 5 Words | Max Score |
|---------|-------------|-----------|
| V1 | `galax`, `rivera`, `world`, `moon`, `moonbeam` | 0.33 |
| V2 | `galax`, `tachygraphy`, `riverside`, `mountainside`, `waterside` | 0.46 |
| V2.1 | `galax`, `tansy`, `taffy`, `riverside`, `mountainside` | 0.47 |
| **V2.2** | **`world`, `earth`, `moon`, `coast`, `water`** | **0.91** |

**Key Observations**:
- V2.2 scores are **2-3x higher** (0.91 vs 0.33-0.47)
- V2.2 returns **semantically meaningful words** instead of token fragments (`galax`) or rare words (`tachygraphy`, `tansy`)
- Sense-aware reranking eliminates polysemy issues
- Diagonal whitening + robust centers create much stronger semantic alignment

**Test: Nature (+1.0) - Urban (-0.5)**

| Version | Top 5 Words | Max Score |
|---------|-------------|-----------|
| V1 | `ocean`, `lake`, `oceanic`, `woods`, `sea` | 0.20 |
| V2 | `lake`, `ocean`, `oceanic`, `sea`, `woods` | 0.16 |
| V2.1 | `lake`, `sea`, `ocean`, `oceanic`, `woods` | 0.16 |
| **V2.2** | **`woods`, `valley`, `sea`, `stream`, `lake`** | **0.29** |

**V2.2 correctly emphasizes wilderness** (`woods`, `valley`, `stream`) over water terms, showing better repulsion from urban concepts.

### Summary of All Improvements

| Improvement | Version | Problem Solved | Impact |
|-------------|---------|----------------|--------|
| **Spherical Averaging** | v2 | High-norm tokens dominate | Better directional alignment |
| **Per-Group Scoring** | v2 | Positive/negative cancellation | Negative weights work as repulsion |
| **All-but-the-Top** | v2 | Embedding anisotropy | Removes common artifacts |
| **Wordfreq Filtering** | v2.1 | Rare/archaic WordNet words | Returns common English words |
| **Tokenization Invariance** | v2.1 | Whitespace tokenization artifacts | Reduces position-dependent quirks |
| **Robust Group Centers** | v2.2 | Outlier entries skew mean | Trimmed mean reduces noise |
| **Sense-Aware Reranking** | v2.2 | Polysemy (multiple word senses) | Uses WordNet glosses for disambiguation |
| **Diagonal Whitening** | v2.2 | Remaining anisotropy | Scales dimensions by inverse std dev |

### CLI Usage

**V2**:
```bash
python word_group_essence_wordnet_v2.py --input groups.json --top-k 20
```

**V2.1**:
```bash
python word_group_essence_wordnet_v21.py --input groups.json --top-k 20 --freq-top-n 200000
```

**V2.2** (recommended):
```bash
python word_group_essence_wordnet_v22.py --input groups.json --top-k 20 --sense-alpha 0.7
```

## API Version Summary

The API now offers 4 versions with progressively better semantic quality:

```
V1 (/compute-essence)
  └─> V2 (/compute-essence-v2)
       └─> V2.1 (/compute-essence-v2-1)
            └─> V2.2 (/compute-essence-v2-2) ⭐ RECOMMENDED
```

**Quick Decision Guide**:
- 🚀 **Want best quality?** → Use `/compute-essence-v2-2`
- ⚡ **Need backwards compatibility?** → Use `/compute-essence` (v1)
- 🔧 **Debugging/comparing?** → Try all versions side-by-side

## Future Enhancements

Potential improvements:
- Support for more language models (Llama, GPT-2, BERT variants)
- Multilingual dictionary support (WordNet equivalents for other languages)
- FAISS-based approximate nearest neighbor search for faster candidate scoring
- Weighted word-group combinations with learned optimal weights
- Integration with contextualized embeddings for comparison
- Package structure for `pip install platonic-ideal`
- Additional v3 improvements: full covariance whitening, CSLS hubness correction, full retrofitting to WordNet edges

## Citation & Acknowledgments

This project uses:
- **Hugging Face Transformers** for tokenizer and model utilities
- **safetensors** for efficient weight loading
- **NLTK WordNet** for English dictionary lemmas
- **Qwen/Qwen2.5-0.5B** model from Alibaba Cloud

## License

(Add your license here)

