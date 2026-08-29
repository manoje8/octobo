
## SECTION 1: RAG FRAMEWORK

### 1.1 Ingestion
 

**Q1. Walk me through your ingestion pipeline for a RAG system handling mixed document types (PDFs, HTML, Confluence pages).**

> Model answer: Use format-specific loaders (PyMuPDF/pdfplumber for PDF, BeautifulSoup or trafilatura for HTML, native API for Confluence) rather than one generic parser, since layout and noise differ a lot by format. Normalize everything into a common intermediate representation (text + structural metadata) before chunking. Extract and preserve metadata at ingestion time — source URL, author, timestamp, section headers, page number — because that metadata becomes filterable/citable at query time and is far cheaper to capture now than to reconstruct later.

  
**Q2. Why does document cleaning matter so much for RAG specifically, more than for training a classifier?**

> Model answer: In RAG, whatever text survives cleaning becomes the literal content injected into the LLM's context and potentially quoted to the user. Boilerplate (headers, footers, nav menus, repeated legal disclaimers) doesn't just add noise to a training signal — it directly pollutes retrieval (matches on boilerplate) and burns context tokens the generator needs for real content. Garbage in isn't just a quality issue here, it's a grounding and hallucination risk.

  

**Q3. How do you handle tables and images during ingestion for a RAG pipeline?**

> Model answer: Tables should generally be serialized in a way that preserves row/column relationships (e.g., markdown table format or explicit key-value flattening) rather than flattened into unstructured prose, since raw text extraction usually scrambles the structure. For images, either run OCR if they contain text-as-image, or generate a caption/description via a vision model and index that description — the raw pixels aren't retrievable by a text embedding model.

  

**Q4. What metadata would you prioritize extracting, and why?**

> Model answer: Source/provenance (for citations and trust), timestamp (for recency filtering and staleness detection), document hierarchy (section/heading path, useful for both chunking boundaries and for showing users where an answer came from), and access-control tags if there's any permissioning requirement. Metadata is what turns "a pile of text" into something you can filter, rank, and audit.

  

---

  

### 1.2 Chunking

  

**Q5. Fixed-size, recursive, and semantic chunking — compare them and say when you'd use each.**

> Model answer: Fixed-size (e.g., 512 tokens with overlap) is simplest and fastest, but it cuts through sentences and ideas arbitrarily, hurting retrieval precision. Recursive chunking splits on a hierarchy of separators (paragraph → sentence → word) so it respects structure while still hitting a target size — a good default for general text. Semantic chunking uses embedding similarity between adjacent sentences to find natural topic boundaries, producing chunks that are more coherent as retrieval units, at the cost of extra compute at ingestion time. I'd default to recursive for most corpora and reach for semantic chunking when the domain has long, topic-dense documents (legal, medical) where a wrong boundary meaningfully changes meaning.

  

**Q6. What's the tradeoff overlap introduces, and how do you pick an overlap size?**

> Model answer: Overlap (e.g., 10-20% of chunk size) protects against splitting a key sentence or fact exactly at a chunk boundary, so the relevant context still appears fully in at least one chunk. The tradeoff is index bloat and duplicate retrieval — the same sentence appearing in multiple chunks can crowd out diverse results in top-k, and it increases storage/embedding cost. I'd tune overlap empirically by measuring recall@k on a held-out QA set at a few overlap settings rather than picking a number theoretically.

  

**Q7. If retrieval precision is low, how would you diagnose whether chunk size is the problem?**

> Model answer: Check whether the correct answer is present anywhere in top-k but in a chunk that's diluted with irrelevant surrounding text (chunk too large) versus the correct chunk not being retrieved at all even though a nearby/related chunk is (chunk too small, losing context needed for a good embedding match). Running the same query set at two or three chunk sizes and comparing recall@k directly is the fastest empirical test — don't just guess from intuition.

  

**Q8. Does chunk size interact with your choice of embedding model?**

> Model answer: Yes — embedding models have a max context and tend to produce better, less "averaged-out" embeddings on focused text. A chunk that's too long relative to what the embedding model handles well gets compressed into a vector that represents a blurry mixture of topics, hurting discrimination between chunks in vector space. So chunk size should be chosen with the embedding model's effective context sweet spot in mind, not just the generator's context window.

  

---

  

### 1.3 Embedding

  

**Q9. How would you choose between text-embedding-3, BGE, and E5 for a given project?**

> Model answer: I'd weigh a few axes: hosted vs. self-hosted (text-embedding-3 is simplest to operate, no infra), MTEB benchmark performance on tasks similar to mine (retrieval vs. clustering vs. classification — BGE and E5 often lead on retrieval-specific benchmarks), licensing and data residency requirements (open-weight models let you self-host for compliance), and cost at scale. In practice I'd shortlist 2-3 candidates and run them through my own retrieval eval (recall@k on a labeled query set) rather than trusting benchmark leaderboards alone, since domain shift matters a lot.

  

**Q10. What's the tradeoff in embedding dimensionality?**

> Model answer: Higher dimensionality generally captures more nuance and separates fine-grained semantic differences better, but it increases storage, index memory, and search latency roughly linearly (or worse, depending on index type). Many modern embedding models (like text-embedding-3) support Matryoshka-style truncation, letting you drop to a lower dimension with a small, controlled accuracy loss — useful when you need to cut costs and can tolerate a slight recall dip.

  

**Q11. When does it make sense to fine-tune an embedding model on domain-specific data, and what does that look like?**

> Model answer: When off-the-shelf embeddings systematically fail to distinguish domain-specific concepts — e.g., legal or medical terms that are near-synonyms in general English but distinct in the domain — fine-tuning pays off. Typically this means contrastive fine-tuning on (query, relevant passage, hard negative) triples mined from your own logs or labeled data, optimizing something like a triplet or InfoNCE loss so domain-relevant passages pull closer together than irrelevant ones. It's worth the investment when you have enough labeled/mined pairs and when general-purpose embeddings are the measured bottleneck, not a hypothetical one.

  

**Q12. How do you evaluate whether an embedding model is a good fit before committing to it in production?**

> Model answer: Build a small labeled eval set of (query, correct passage) pairs from real or representative use cases, then measure recall@k and MRR for each candidate model on that set — not just relying on published MTEB scores, which may not reflect your domain. I'd also sanity-check qualitatively: pull a handful of near-miss retrievals and see if they're "reasonably related but wrong" (embedding is working, retrieval logic needs tuning) versus "unrelated" (embedding itself is the problem).

  

---

  

### 1.4 Retrieval

  

**Q13. Dense vs. sparse (BM25) vs. hybrid retrieval — when does each win?**

> Model answer: Dense retrieval (embeddings) wins on semantic/paraphrase matches — queries and documents that mean the same thing but use different words. Sparse retrieval (BM25) wins on exact-term matches — IDs, product codes, rare proper nouns, acronyms — where dense embeddings can blur distinctions that keyword matching nails precisely. Hybrid search (combining both, typically via reciprocal rank fusion or a weighted score blend) is usually the most robust default in production because it covers both failure modes, at the cost of extra complexity and compute.

  

**Q14. How do you choose top-k, and what happens if you set it too high or too low?**

> Model answer: Top-k too low risks missing a relevant passage entirely (hurts recall, and there's no reranker downstream to recover it). Top-k too high dilutes the generator's context with marginally relevant or irrelevant chunks, which increases hallucination risk and cost, and can bury the truly relevant chunk in the middle of a long context (the "lost in the middle" effect). I'd tune k empirically against recall@k curves — pick the smallest k that captures acceptable recall — and treat k differently depending on whether a reranker sits downstream (if so, retrieve a wider net, e.g. k=50, then rerank down to k=5).

  

**Q15. How does hybrid search actually combine dense and sparse scores in practice?**

> Model answer: Common approaches are Reciprocal Rank Fusion (RRF), which combines rankings rather than raw scores (robust because it avoids needing to normalize incomparable score scales), or a weighted linear combination of normalized dense and BM25 scores. RRF is often preferred in practice because dense cosine similarity and BM25 scores live on very different scales and weighting them directly is fragile without careful calibration.

  

---

  

### 1.5 Reranking

  

**Q16. Why add a cross-encoder reranker if you already have a good retriever?**

> Model answer: Bi-encoders (used for the initial dense retrieval) embed the query and document independently, which is fast and scalable but loses the ability to model fine-grained query-document interaction. A cross-encoder reranker feeds the query and document together through the model, letting attention directly compare them — much more accurate but too slow to run over the full corpus. So the standard pattern is: cheap bi-encoder retrieves a wide candidate set (e.g. top-50), then the expensive cross-encoder reranks that small set down to the true top-k, getting both speed and precision.

  

**Q17. What's the cost/latency tradeoff of adding a reranking stage, and how do you decide if it's worth it?**

> Model answer: Reranking adds a model inference pass over every candidate (e.g., 50 cross-encoder calls), which adds real latency, especially without batching or a hosted reranker API. It's worth it when retrieval precision genuinely lags — e.g., recall@50 is good but precision@5 or nDCG is weak — meaning the right document is in the candidate pool but not surfaced near the top. I'd measure precision@k before and after adding a reranker on a labeled eval set to justify the added latency rather than assuming it always helps.

  

---

  

### 1.6 Generation

  

**Q18. How do you construct a prompt to maximize grounding and minimize hallucination in RAG?**

> Model answer: Explicitly instruct the model to answer only from the provided context and to say it doesn't know rather than guess when the context is insufficient. Structure retrieved chunks clearly (with source labels/IDs) so the model can attribute claims to specific passages. Put the most relevant chunk near the start or end of the context rather than buried in the middle, since models attend less reliably to the middle of long contexts. And keep irrelevant chunks out of the context in the first place — the prompt can't fix bad retrieval.

  

**Q19. What citation strategies have you used to reduce hallucination, and what are their tradeoffs?**

> Model answer: Inline citation tags where the model is prompted to tag each claim with a source ID (e.g., [1], [2]) mapped back to retrieved chunks — this makes hallucinated, uncited claims visible to the user, though the model can still fabricate a citation-looking tag on an unsupported claim, so it's a mitigation, not a guarantee. A stronger but more expensive approach is a post-hoc verification pass: after generation, check whether each sentence is actually entailed by a cited passage (using a smaller NLI-style model), and flag or strip unsupported sentences before returning the answer.

  

**Q20. If a user reports a hallucinated answer, how do you determine whether it's a generation problem versus a retrieval problem?**

> Model answer: Check whether the correct information was actually present anywhere in the retrieved context that was fed to the generator. If it wasn't retrieved at all, it's a retrieval failure — no amount of prompt engineering fixes that. If it was retrieved but the model still generated something unsupported or wrong, it's a generation/grounding failure — the fix is prompt tightening, adding citation-forcing, or a verification pass, not touching the retriever.

  

---

  

### 1.7 Evaluation

  

**Q21. Walk me through recall@k and MRR — what do they measure and when would you prefer one over the other?**

> Model answer: Recall@k measures whether at least one relevant document appears in the top k results — good for asking "did we find it at all." MRR (Mean Reciprocal Rank) measures how high up the first relevant result appears, averaged across queries — good for asking "did we find it near the top," which matters more when the generator or user will only look closely at the first result or two. I'd track recall@k to catch total retrieval misses, and MRR (or nDCG for graded relevance) to catch ranking quality issues even when recall is technically fine.

  

**Q22. How do RAGAS-style faithfulness and answer relevance metrics work, and what are their limitations?**

> Model answer: Faithfulness typically decomposes the generated answer into individual claims and checks, often using an LLM-as-judge, whether each claim is entailed by the retrieved context — a proxy for hallucination. Answer relevance checks how well the answer addresses the actual question, sometimes by generating a hypothetical question from the answer and measuring its similarity to the original query. The main limitation is that these are themselves LLM-judged metrics, so they inherit judge-model biases and aren't perfectly reliable ground truth — I'd treat them as a scalable proxy to complement, not replace, periodic human evaluation.

  

**Q23. How would you build an evaluation set for a RAG system from scratch when you have no labeled data?**

> Model answer: Start by mining real user queries (if any exist) or synthetically generating question-answer-passage triples from the corpus using an LLM, then have a human spot-check a sample for quality. Stratify the eval set across query types you expect in production (factual lookup, multi-hop, out-of-scope) so the metric isn't dominated by the easy case. Keep it small but curated (50-200 well-checked examples) rather than large but noisy — evaluation quality matters more than volume early on.

  

---

  

### 1.8 Failure Modes (Diagnostic / Scenario Questions)

  

**Q24. A user asks a question and gets a confidently wrong answer. Walk me through how you'd diagnose whether it's a retrieval miss, a hallucination, or a stale index.**

> Model answer: First, log and inspect exactly what chunks were retrieved for that query. If the correct information isn't in any retrieved chunk, that's a retrieval miss — dig into whether it's an embedding mismatch, a chunking issue (right doc, wrong split), or a genuine indexing gap. If the correct information is present in the retrieved context but the answer still contradicts it, that's a generation/grounding failure. If the retrieved chunk contains outdated information that used to be correct, that's a stale index — check the document's last-updated timestamp and your reindexing cadence. Having per-query retrieval logs (query, retrieved chunk IDs, generated answer) is what makes this diagnosis possible at all — without it you're guessing.

  

**Q25. Your RAG system was working well in testing but degrading over time in production. What's your first hypothesis, and how do you test it?**

> Model answer: First hypothesis is index staleness — the underlying documents changed but the index wasn't refreshed, or new documents were never ingested. I'd check reindexing job logs and compare document timestamps in the source system against the last index update. Second hypothesis is query distribution drift — production queries look different from the eval set they tested against, so I'd sample recent production queries and manually check retrieval quality on those specifically rather than trusting the old eval set.

  

**Q26. How would you design monitoring to catch these failure modes before users report them?**

> Model answer: Track retrieval confidence/similarity scores as a leading indicator — a sudden drop in average top-1 similarity across queries can signal drift or an indexing problem. Track a sampled faithfulness metric (RAGAS or similar) continuously, not just at launch. Track index freshness explicitly (time since last successful reindex, count of documents pending ingestion). And keep a canary set of known question-answer pairs that get re-run on a schedule to catch regressions early.

  

---

  

## SECTION 2: VECTOR DATABASES & AI PIPELINES

  

### 2.1 Vector DBs & Index Types

  

**Q27. Compare FAISS, Pinecone, Weaviate, Chroma, and Milvus — how would you choose among them?**

> Model answer: FAISS is a library, not a managed service — great for full control and cost efficiency if you're willing to run and scale it yourself. Pinecone is fully managed, low ops burden, good for teams that want to move fast without infra work. Weaviate and Milvus are open-source with managed cloud options, and both support hybrid search and richer metadata filtering out of the box, which matters if filtering is a core requirement. Chroma is lightweight and great for prototyping or smaller-scale local-first apps but less proven at very large scale. I'd choose based on team's ops appetite, scale requirements, and whether built-in hybrid search/filtering saves meaningful engineering time.

  

**Q28. Explain HNSW, IVF, and flat (brute-force) index types, and the tradeoffs between them.**

> Model answer: Flat/brute-force computes exact similarity against every vector — perfectly accurate but scales linearly with corpus size, fine for small collections (tens of thousands of vectors) but too slow at scale. IVF (Inverted File Index) clusters vectors and only searches the nearest clusters at query time, trading a small accuracy loss for a big speed gain, with a tunable number of clusters to probe. HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity graph and is currently the most common choice for approximate nearest neighbor search — very fast query times and strong recall, at the cost of higher memory usage and slower index build times than IVF.

  

**Q29. When is approximate nearest neighbor search "acceptable," and when would you insist on exact search?**

> Model answer: Approximate search is acceptable — and usually necessary — whenever corpus size makes brute-force too slow for the latency budget, which in practice means most production RAG systems above a few hundred thousand vectors. I'd insist on exact search only for small collections where brute-force is already fast enough, or in domains with very low tolerance for missing the single best match (e.g. some compliance or safety-critical lookups) where even a small recall drop from approximation is unacceptable — and even then I'd first measure whether the approximation actually causes real-world misses before paying the cost of exact search.

  

**Q30. How do you tune an HNSW index in practice — what parameters matter most?**

> Model answer: `M` controls the number of connections per node in the graph — higher M improves recall but increases memory and build time. `ef_construction` controls search thoroughness during index building — higher values build a better-quality graph at the cost of slower indexing. `ef_search` controls thoroughness at query time — higher values improve recall at the cost of query latency, and it's the easiest one to tune post-hoc without rebuilding the index. I'd start with common defaults (M=16, ef_construction=200) and tune ef_search against a recall/latency curve for the specific workload.

  

---

  

### 2.2 Data Labeling & Active Learning

  

**Q31. When would you use an active learning loop instead of just labeling a random sample of data?**

> Model answer: Active learning is worth it when labeling is expensive (specialist annotators, complex domain judgments) and the model can meaningfully tell you which unlabeled examples it's most uncertain about. Rather than labeling randomly, you label the examples nearest the model's decision boundary (highest uncertainty, or highest expected model change), which typically gets you to a target accuracy with substantially fewer labels than random sampling. It's less valuable when labeling is cheap and fast, or when the data distribution is simple enough that random sampling already covers it well.

  

**Q32. Compare Label Studio, Prodigy, and SageMaker Ground Truth — what would drive your choice?**

> Model answer: Label Studio is open-source, highly configurable across many data/task types, good if you want full control and no per-seat licensing cost. Prodigy is a paid, developer-first tool built specifically around efficient active-learning-driven annotation workflows (particularly strong for NLP tasks), good when annotation efficiency and tight active-learning loops matter more than avoiding license cost. SageMaker Ground Truth is the managed AWS option, useful when you're already deep in the AWS ecosystem and want built-in workforce management (including access to Mechanical Turk-style crowd labor) without standing up your own tooling.

  

**Q33. How do you decide when an active learning loop has "converged" and you can stop labeling?**

> Model answer: Track model performance on a fixed held-out validation set as each active learning batch comes in, and stop when the marginal improvement per additional batch of labels flattens out (diminishing returns on the learning curve). I'd also sanity check that the uncertainty distribution of remaining unlabeled data has shifted — if the model is no longer finding many high-uncertainty examples, that's a sign it's reached the limits of what more labeling on this distribution can offer.

  

---

  

### 2.3 Pipeline Orchestration

  

**Q34. When would you reach for Airflow/Prefect versus LangChain/LlamaIndex chains?**

> Model answer: Airflow/Prefect are general-purpose batch/ETL orchestrators — good for scheduled, DAG-based jobs like nightly reindexing, data ingestion pipelines, or retraining jobs, where you need retries, monitoring, and dependency management across heterogeneous tasks. LangChain/LlamaIndex chains are purpose-built for GenAI request-time flows — chaining retrieval, reranking, and generation within a single user-facing request. In a mature system I'd expect both: Airflow orchestrating the offline ingestion/embedding/indexing pipeline on a schedule, and a LangChain/LlamaIndex-style chain (or custom code) handling the online query-time RAG flow.

  

**Q35. What can go wrong if you try to use a request-time chaining framework for what should really be a batch orchestration job, or vice versa?**

> Model answer: Using a request-time framework for batch work loses you retries, backfills, scheduling, and dependency graphs that a real orchestrator gives you for free — you end up half-reimplementing Airflow badly. Using a heavy batch orchestrator for request-time chaining adds latency and operational overhead unsuited to a single low-latency user request — DAG scheduling overhead doesn't belong in the hot path. The rule of thumb: offline/scheduled/bulk work goes in the orchestrator, online/low-latency/per-request work goes in the chain framework.

  

---

  

## SECTION 3: RESPONSIBLE AI / FAIRNESS

  

### 3.1 Fairness Metrics

  

**Q36. Explain demographic parity and equalized odds, and why they can't both be satisfied in general.**

> Model answer: Demographic parity requires the positive prediction rate to be equal across groups, regardless of the true outcome. Equalized odds requires the true positive rate and false positive rate to be equal across groups, conditioned on the actual outcome. These can conflict whenever the base rates of the true outcome differ across groups — satisfying equal prediction rates (demographic parity) while base rates differ mathematically forces unequal error rates between groups (violating equalized odds), and vice versa. This is a known impossibility result, not an engineering oversight, so the real conversation in practice is which fairness definition is appropriate for the specific decision being made, not "figuring out how to hit both."

  

**Q37. If you can't satisfy both demographic parity and equalized odds, how do you decide which one to prioritize?**

> Model answer: It depends on the harm model of the decision. If false negatives are the primary harm (e.g., a medical screening test missing disease), equalized odds — specifically equal true positive rates — is usually the more defensible target. If the concern is about representation/access regardless of underlying differences in the population (e.g., resume screening where you want proportional advancement), demographic parity may be the better fit. This is ultimately a policy/values decision informed by domain context, not a purely technical optimization — I'd want that decision made explicitly and documented, not implicitly baked into a metric choice.

  

**Q38. What's the difference between individual fairness and group fairness metrics?**

> Model answer: Group fairness metrics (demographic parity, equalized odds) look at aggregate statistics across protected groups. Individual fairness asks a different question — that similar individuals should receive similar predictions, regardless of group membership — which requires defining a similarity metric between individuals, itself a nontrivial and value-laden choice. Group fairness is more common in practice because it's easier to measure and audit at scale, but it can permit unfairness at the individual level even when group-level numbers look balanced.

  

---

  

### 3.2 Bias Sources

  

**Q39. Walk through the main sources of bias in an ML pipeline — training data, labels, and features.**

> Model answer: Training data imbalance occurs when certain groups are underrepresented in the data, so the model has less signal to learn accurate patterns for them. Label bias occurs when the ground truth labels themselves reflect historical human bias (e.g., historical hiring decisions used as labels for a hiring model bake in whatever bias existed in those past decisions). Proxy variables are features that aren't explicitly a protected attribute but correlate strongly with one (e.g., zip code correlating with race), so a model can reconstruct discriminatory patterns even if the protected attribute itself is excluded from the model.

  

**Q40. If you remove protected attributes from your training data ("fairness through unawareness"), does that solve the bias problem?**

> Model answer: No — this is a common but insufficient approach, because proxy variables can let the model reconstruct the same discriminatory signal indirectly. In some cases, removing the protected attribute actually makes bias harder to detect and correct, since you can no longer directly measure group-level disparities without it. A more robust approach usually keeps the protected attribute available for auditing purposes (measuring fairness metrics) even while excluding it, or excluding it, from the actual prediction features.

  

**Q41. How would you detect that a proxy variable is causing disparate impact, if the protected attribute itself isn't in the model?**

> Model answer: Measure outcome disparities across protected groups directly (using held-out demographic data even if it's not a model feature), and if disparities show up, do a feature-importance or correlation analysis to identify which input features correlate most strongly with group membership. SHAP or similar attribution methods can help pinpoint which features are driving disparate predictions for different groups, pointing toward likely proxy variables.

  

---

  

### 3.3 Mitigation

  

**Q42. Compare re-sampling, re-weighting, adversarial debiasing, and post-processing threshold adjustment as bias mitigation strategies.**

> Model answer: Re-sampling (oversampling underrepresented groups or undersampling overrepresented ones) and re-weighting (giving underrepresented examples more weight in the loss) are both pre-processing techniques applied to training data, simple to implement but limited to what data imbalance alone can fix. Adversarial debiasing trains the main model alongside an adversary trying to predict the protected attribute from the model's internal representation, penalizing the main model when the adversary succeeds — a more powerful in-processing technique but more complex to train and tune. Post-processing threshold adjustment sets different decision thresholds per group after the model is already trained — cheapest to implement and doesn't require retraining, but is the least addressing of root causes and can raise its own fairness/legal questions since it explicitly treats groups differently at decision time.

  

**Q43. If you apply post-processing threshold adjustment to equalize outcomes across groups, what tradeoff or risk does that introduce?**

> Model answer: You're explicitly using group membership at decision time to set different bars, which can face legal and ethical objections (disparate treatment concerns, versus the disparate impact concerns it's meant to fix), and it only fixes the symptom at the final decision point without addressing why the underlying model scores differ across groups in the first place. It's often used as a fast tactical fix while a more fundamental data or model fix is developed, not as a permanent solution.

  

**Q44. How would you decide which mitigation technique to use for a given project?**

> Model answer: I'd start by diagnosing where the bias is actually coming from — if it's primarily data imbalance, re-sampling/re-weighting is the direct fix; if the model is learning to exploit proxy signals even with balanced data, in-processing techniques like adversarial debiasing address it more fundamentally; if you need a fast fix without retraining (e.g., a model already in production), post-processing is the pragmatic stopgap while a longer-term fix is built. I'd also factor in regulatory/legal constraints, since some jurisdictions restrict using protected attributes even for adjustment purposes.

  

---

  

### 3.4 Explainability

  

**Q45. Compare SHAP and LIME — how do they work and what are the tradeoffs?**

> Model answer: LIME explains an individual prediction by perturbing the input locally and fitting a simple, interpretable model (like linear regression) to approximate the complex model's behavior in that local neighborhood — fast but the explanation quality depends heavily on how the local neighborhood is sampled and can be unstable across runs. SHAP is grounded in cooperative game theory (Shapley values), attributing a prediction's outcome fairly across features based on their marginal contribution across all possible feature coalitions — more theoretically principled and consistent, but more computationally expensive, especially for models without an efficient SHAP approximation (like TreeSHAP for tree-based models).

  

**Q46. How do you make an LLM-based system explainable, given that SHAP/LIME are hard to apply directly to generation?**

> Model answer: Attention visualization can show which input tokens the model weighted most heavily when generating an output, though it's a weaker and less reliable explanation than SHAP/LIME since attention weights don't always correspond cleanly to causal importance. In practice, citation-grounding is the more practical and trustworthy explainability tool for RAG/LLM systems — if every claim in the answer is tied back to a specific retrieved source passage, the user (and you, when debugging) can directly verify what the answer is based on, which is a more actionable form of explainability than trying to interpret internal model weights.

  

**Q47. A stakeholder asks "why did the model make this specific prediction/answer?" for a RAG-based system. What do you actually show them?**

> Model answer: For a RAG system specifically, I'd show the retrieved source passages that were used to generate the answer, with inline citations mapping each claim in the answer back to a specific passage — this is usually the most concrete and trustworthy explanation available, more so than trying to explain the generator's internal weights. For a traditional classifier feeding into the same system (e.g., a routing or ranking model), I'd supplement with SHAP values on the top features driving that specific decision.

  

---

  

## RAPID-FIRE ROUND (quick recall check)

  

Answer each in one sentence, then check.

  

1. **What's the "lost in the middle" problem?** → LLMs attend less reliably to information placed in the middle of a long context versus the start or end.

2. **What does nDCG add over recall@k?** → It accounts for the *position* and *graded relevance* of results, not just whether a relevant one appears somewhere in top-k.

3. **Why is BM25 still relevant when we have embeddings?** → It excels at exact keyword/entity matches (IDs, rare terms) that dense embeddings can blur.

4. **What's a hard negative in embedding fine-tuning?** → A passage that's superficially similar to the query but actually irrelevant, used to sharpen the model's discrimination.

5. **What does "faithfulness" measure in RAGAS?** → Whether each claim in the generated answer is actually supported by the retrieved context.

6. **Why can demographic parity and equalized odds conflict?** → Because they can't both hold simultaneously when base rates of the true outcome differ across groups.

7. **What's a proxy variable?** → A feature correlated with a protected attribute that lets a model reconstruct discriminatory signal indirectly.

8. **HNSW vs. IVF — which typically has better recall at the cost of more memory?** → HNSW.

9. **What's the first thing to check when a RAG answer is wrong?** → Whether the correct information was in the retrieved context at all (retrieval miss) or was present but ignored/contradicted (generation failure).

10. **Why keep the protected attribute available for auditing even if it's excluded from model features?** → So you can still measure group-level fairness metrics; removing it entirely makes bias harder to detect.

  

---

  

*Good luck — if you want, tell me which section felt shakiest and I can generate a second, harder pass focused just on that area, or turn any of these into a longer whiteboard-style discussion.*