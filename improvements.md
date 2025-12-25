**IMPLEMENTATION STATUS**: 
- **v2** (improvements #1-3): `/compute-essence-v2` - Spherical averaging, per-group scoring, All-but-the-Top
- **v2.1** (improvement #6): `/compute-essence-v2-1` - Adds tokenization invariance + wordfreq filtering
- **v2.2** (improvements #4, #5, #8): `/compute-essence-v2-2` - Adds diagonal whitening, trimmed mean, sense-aware reranking

**RECOMMENDED**: Use `/compute-essence-v2-2` for production - it delivers dramatically better semantic quality (scores 2-3x higher, much more meaningful words).

---

From the README, the repo's "word prediction" is basically:

* take the model's **token embedding matrix** (E\in\mathbb{R}^{V\times D}),
* build an **entry vector** by averaging token embeddings for each input word/phrase,
* average entries into **group means**, apply **group weights**, average groups into one **essence vector**,
* then return the **nearest WordNet lemma(s)** by **cosine similarity**. ([GitHub][1])

That’s a clean linear pipeline. The main ways to improve “accuracy” (i.e., the semantic quality of the nearest-word retrieval) are to (1) improve the estimator for the essence vector, and (2) use a better geometry/metric than raw cosine on raw token embeddings.

Below are mathematically grounded upgrades that tend to give noticeable gains in nearest-neighbor word retrieval, while keeping your overall approach intact.

---

## 1) Normalize *before* you average (spherical mean instead of Euclidean mean) ✅ IMPLEMENTED

Right now your entry vector is (conceptually) the Euclidean mean:
[
v_e=\frac1{T_e}\sum_{j=1}^{T_e}E[t_j].
] ([GitHub][1])

But token vectors have different norms (often correlated with frequency/artifacts). If you average unnormalized vectors, high-norm tokens can dominate.

A very common fix is: **L2-normalize each token embedding first**, then average, then normalize again:
[
\hat E[t]=\frac{E[t]}{\lVert E[t]\rVert_2},\qquad
v_e=\operatorname{norm}!\left(\frac1{T_e}\sum_{j=1}^{T_e}\hat E[t_j]\right),
]
where (\operatorname{norm}(x)=x/|x|_2).

Do the same at group level:
[
m_g=\operatorname{norm}!\left(\frac1{N_g}\sum_{e\in g} v_e\right).
]

Why this helps (geometrically): you’re averaging **directions** on the unit sphere, which matches cosine-based retrieval better than averaging raw vectors in (\mathbb{R}^D).

---

## 2) Don't let positives + negatives "cancel" into a weak direction ✅ IMPLEMENTED

Your README describes negative weights as “repulsion” and positives as “attraction.” ([GitHub][1])
But if you collapse everything into a single essence vector
[
V=\frac1G\sum_g w_g,m_g
]
then **positive and negative groups can cancel**, producing a small or “generic” direction. Then cosine nearest neighbors can drift into weird regions.

A more stable scoring rule is to **score each candidate word against each group separately**, then combine scores:
[
\text{score}(w)=\sum_g w_g;\cos!\big(\hat v_w,\hat m_g\big).
]

This is still very “linear algebra,” but it avoids the worst cancellation pathologies. It also makes “repulsion” literal: a word gets penalized if it points toward a negative group.

Practical note: this is easy to implement because you already compute (m_g) and cosine similarities. You just compute multiple cosines instead of one.

---

## 3) Fix anisotropy: subtract mean + remove top principal components ("All-but-the-Top") ✅ IMPLEMENTED

Static embedding spaces are often **anisotropic**: many vectors share a big common component, so cosine similarity gets inflated in “uninteresting” directions. This hurts nearest-neighbor quality.

A simple, very effective postprocess:

1. Build a matrix (X\in\mathbb{R}^{N\times D}) of vectors you care about (e.g., all WordNet candidate vectors).
2. Compute the mean (\mu=\frac1N\sum_i X_i).
3. Compute top (k) principal directions (U_k\in\mathbb{R}^{D\times k}) of ((X-\mu)).
4. Transform any vector (v) by:
   [
   v'=(I-U_kU_k^\top),(v-\mu).
   ]

Then do cosine search using (v') for both candidates and the essence vector.

This is exactly the “remove mean + remove top directions” method shown to improve word similarity/analogy tasks across embeddings. ([arXiv][2])

If you implement just one “math upgrade,” implement this.

---

## 4) Whitening / Mahalanobis geometry instead of raw cosine ✅ IMPLEMENTED (v2.2: diagonal whitening)

A step beyond #3: use a **whitening transform** based on the candidate distribution.

Let (\Sigma) be the covariance of candidate vectors (after centering). Define:
[
W=\Sigma^{-1/2}.
]
Transform vectors (v\mapsto W(v-\mu)), then use cosine similarity in the whitened space.

Intuition: cosine in whitened space behaves like a **Mahalanobis-aware similarity**, downweighting high-variance directions that otherwise dominate retrieval.

Whitening has repeatedly been shown to alleviate embedding anisotropy and improve semantic similarity. ([ar5iv][3])

---

## 5) Use a robust estimator for group centers (geometric median / trimmed mean) ✅ IMPLEMENTED (v2.2: trimmed mean)

Averages are fragile: one weird entry (“earthporn”, artifacts, polysemy, tokenization weirdness) can yank the mean.

A robust replacement is the **geometric median**:
[
m_g=\arg\min_{v\in\mathbb{R}^D}\sum_{e\in g}\lVert v-v_e\rVert_2.
]

This is “just optimization,” but it often improves stability when groups contain noisy words.

A cheaper approximate alternative: **trimmed mean**:

* compute preliminary mean,
* drop the bottom (p%) entries by cosine-to-mean,
* recompute the mean.

This pairs nicely with your existing “input validation / filtering” philosophy. ([GitHub][1])

---

## 6) Tokenization invariance trick (still non-contextual, but much better) ✅ IMPLEMENTED (v2.1)

Many tokenizers represent a word differently depending on whitespace / position. If you embed the isolated string `"apple"` you may get different tokens than `" apple"` in running text.

A simple way to reduce this:

[
v_w=\frac12\left(\operatorname{mean}(E[\text{tok}(w)]);+;\operatorname{mean}(E[\text{tok}(\text{" "}+w)])\right)
]
(and then normalize).

You’re still only using the embedding matrix (no forward pass), but you get something closer to how the model “usually” sees that word.

This matters a lot for nearest-word quality and is completely consistent with your “platonic embedding matrix” setup. ([GitHub][1])

---

## 7) Retrofitting to WordNet (perfectly aligned with your candidate set)

Since you *already* restrict candidates to WordNet lemmas, you can improve “accuracy” by adjusting vectors to respect WordNet edges.

Classic “retrofitting” formulation:

Given original vectors (z_i) (your candidate word vectors) and a WordNet graph (E) (edges like synonymy, etc.), solve:
[
\min_{{q_i}};
\sum_i \alpha_i\lVert q_i-z_i\rVert_2^2
;+;
\sum_{(i,j)\in E}\beta_{ij}\lVert q_i-q_j\rVert_2^2.
]

This pulls related WordNet words closer while keeping vectors near the originals. It’s a pure quadratic objective; the standard solution is an efficient iterative update.

This method was introduced exactly for “improve lexical semantic quality using WordNet / lexicons,” and tends to improve nearest-neighbor semantic tasks. ([arXiv][4])

In your setting it’s especially natural because:

* your retrieval dictionary *is* WordNet,
* you can retrofit only the candidate vectors (offline), then keep the rest of the pipeline identical.

---

## 8) Sense-aware WordNet ranking (polysemy is a big hidden error source) ✅ IMPLEMENTED (v2.2: gloss-based reranking)

WordNet lemmas are polysemous (“bank”, “bat”, “seal”…). A single lemma vector is a mixture of senses, so nearest-neighbor can look “wrong.”

A math-only fix that still uses your same embedding machinery:

* For each synset (s), embed its gloss/definition text into a vector (g_s) (using the same token-mean method).
* Define a **sense vector**:
  [
  u_s=\operatorname{norm}\big(\alpha,v_{\text{lemma}(s)} + (1-\alpha),g_s\big)
  ]
* Score synsets by (\cos(u_s, V_{\text{essence}})), pick the best synset, and return its lemma.

This tends to make predictions “feel” far more accurate because you’re matching the intended sense, not the blended lemma.

(Still just linear combinations + cosine.)

---

## 9) Add an actual quantitative “accuracy” loop (so you can tune (k,\alpha), etc.)

Right now the repo gives qualitative guidance (“add more entries,” “adjust weights,” etc.). ([GitHub][1])
To improve accuracy *systematically*, define a measurable proxy task:

Example: **WordNet synonym retrieval**

* For each synset, take one lemma as “query”.
* Build an input group from the other lemmas in that synset.
* Run your pipeline to predict top-(K) words.
* Count a hit if any lemma from the held-out synset appears in top-(K).

Metrics:

* top-(K) accuracy
* mean reciprocal rank (MRR)

Then tune:

* PC-removal (k) (often 1–10),
* whitening on/off,
* spherical mean on/off,
* gloss-mix (\alpha),
* per-group scoring vs single-essence-vector scoring.

This turns “accuracy improvement” into an optimization problem instead of vibes.

---

## What I’d implement first (highest ROI)

1. **All-but-the-top** (mean subtraction + remove top PCs) on candidate vectors + essence vector. ([arXiv][2])
2. **Spherical averaging** (normalize then average) at token→entry and entry→group steps. ([GitHub][1])
3. Switch scoring to **(\sum_g w_g\cos(v_w,m_g))** rather than collapsing to one vector (especially when using negative weights). ([GitHub][1])
4. If you want to go further: **retrofitting** to WordNet. ([arXiv][4])

If you tell me which failure mode you see most often (polysemy? tokenization fragments? negatives not working? results too “generic”?) I can point you to the *smallest* mathematical change that targets that specific issue—but the four above are the most generally effective for this exact pipeline.

[1]: https://github.com/Hilo-Hilo/platonic-ideal "GitHub - Hilo-Hilo/platonic-ideal: a project to use large language model tokenizers and embedding matriies to 'extract the essence' of a given list or words, and return the most similar token match to the 'essence''"
[2]: https://arxiv.org/abs/1702.01417?utm_source=chatgpt.com "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"
[3]: https://ar5iv.labs.arxiv.org/html/2103.15316?utm_source=chatgpt.com "Whitening Sentence Representations for Better Semantics ..."
[4]: https://arxiv.org/abs/1411.4166?utm_source=chatgpt.com "Retrofitting Word Vectors to Semantic Lexicons"
