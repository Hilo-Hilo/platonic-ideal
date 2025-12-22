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

### 1. Create Virtual Environment
```bash
cd /home/hansonwen/platonic-ideal
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dependencies:
- `huggingface_hub` - Download model files from Hugging Face
- `safetensors` - Load embedding weights
- `transformers` - Tokenizer
- `numpy` - Vector operations
- `nltk` - WordNet lexical database
- `torch` - Handle bfloat16 tensors (CPU-only)

### 3. Download WordNet Data (one-time)
```bash
python - <<'PY'
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("WordNet data downloaded successfully")
PY
```

## Quick Start

### Extract Token Embeddings

Get the embedding vector for a specific token ID:
```bash
python extract_embeddings.py --token-id 100
```

Get embeddings for text (shows tokenization + vectors):
```bash
python extract_embeddings.py --text "Hello world"
```

### Compute Word-Group Essence

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

## Backend API (FastAPI)

This repository includes a minimal backend service that exposes the essence computation as JSON APIs. It **loads models on-demand per request** (no RAM cache) and defaults to the best-performing model from tests: **TinyLlama/TinyLlama-1.1B-Chat-v1.0**.

### Endpoints
- `GET /health` — health check, returns available models
- `POST /compute-essence` — main endpoint: compute essence and nearest WordNet words

### Session concurrency + model limits (scalability)
To keep the service scalable and prevent a single user (or many browser tabs) from overloading the backend:

- **One in-flight request per user session**: the frontend sends an `X-Session-ID` header (stored in `localStorage`, shared across tabs). The backend rejects concurrent requests for the same session with **HTTP 429**.
- **Max 3 models per request**: the backend enforces `len(model_ids) <= 3` and the UI prevents selecting more than 3 models at once.

Locking backend implementation:
- If `REDIS_URL` is set, the backend uses a **Redis distributed lock** (recommended for multiple instances).
- Otherwise it falls back to an in-process lock (OK for local dev).

### Run locally (dev)
```bash
source venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
```

> If the frontend shows “Cannot connect to backend” but `curl http://localhost:8001/health` works, your browser may be resolving `localhost` to IPv6 (`::1`) while `uvicorn --host 0.0.0.0` only listens on IPv4.  
> Fix: use `http://127.0.0.1:8001` (the frontend defaults to this), or run uvicorn with an IPv6 host (e.g. `--host ::`) if that fits your environment.

### Docker (production-style)
```bash
docker build -t platonic-ideal-api .
docker run -p 8001:8001 platonic-ideal-api
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

### Model IDs available
- `tinyllama-1.1b` (default) → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `qwen-0.5b` → `Qwen/Qwen2.5-0.5B`
- `qwen-1.5b` → `Qwen/Qwen2.5-1.5B`
- `qwen-3b` → `Qwen/Qwen2.5-3B`
- `qwen-7b` → `Qwen/Qwen2.5-7B`
- `phi-2` → `microsoft/phi-2`
- `gemma-2b` → `google/gemma-2b`

> Note: Models are loaded on-demand per request. TinyLlama runs in ~1.8s; larger models can take >2 minutes. For production, consider caching if acceptable, but this API intentionally avoids persistent RAM cache per your requirement.

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

## Files in This Repository

### Core Scripts
- **`extract_embeddings.py`** - Base functionality: download and load embedding matrix, extract token vectors
- **`word_group_essence_wordnet.py`** - Main method: compute word-group essence and find nearest dictionary words

### Configuration Files
- **`requirements.txt`** - Python package dependencies
- **`groups.json`** - Example input (user-editable)

### Output Files (generated)
- **`groups.out.json`** - Computed results with top words
- **`groups.wordnet.out.json`** - Same (if you use different output names)

### Infrastructure
- **`venv/`** - Python virtual environment (gitignored)
- **`.gitignore`** - Excludes venv, cache, temporary files

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

