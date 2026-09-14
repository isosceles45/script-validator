# TFS Creative Script Validator

Scores a creator-submitted script against a campaign brief and the TFS product
manual corpus, on three axes:

| Axis | What it measures | How it is scored |
|---|---|---|
| **Brief alignment** | Does the script deliver the objective, audience, tone, key message and mandatories? | LLM rubric, 5 dimensions |
| **Marketing message quality** | Is it good copy? Hook, clarity, structure, persuasiveness, CTA, brand voice | LLM rubric, 6 dimensions |
| **Product claim validity** | Is every factual claim backed by the manuals? | RAG + deterministic penalty table |

Output is a verdict (`approved` / `approved_with_edits` / `needs_revision` /
`needs_revision_blocking`), per-claim verdicts with manual citations, a reviewer
note, and the retrieval evaluation for that run.

---

## 1. Setup

Requires Python 3.12+ and an API key for **either** OpenAI or Gemini.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env: set PROVIDER and the matching API key
```

Minimum viable `.env`:

```ini
PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Verify without spending anything — the test suite runs fully offline against a
deterministic fake embedder:

```bash
make test        # 56 tests, no API key, no network
```

## 2. Add the manuals

Drop the product manuals into `data/manuals/`. Supported: `.pptx`, `.pdf`,
`.docx`, `.txt`, `.md`.

```
data/manuals/
├── [Training] TFS_Tea_Tree_Pore_Ampoule_202207.pptx
├── ENGTFS_Vita_Drop_Sunquid_2403.pptx
└── ...
```

Legacy `.ppt` and `.doc` are **not** readable by any Python library. Convert them
first — ingestion reports them by name rather than skipping silently:

```bash
soffice --headless --convert-to pptx --outdir data/manuals data/manuals/*.ppt
```

## 3. Ingest

```bash
make ingest                              # or: .venv/bin/python -m app.cli ingest
```

Takes ~75s and about 1.5 cents for the 42-manual TFS corpus. Idempotent by file
checksum — re-running only processes manuals that changed. Use `--force` after
changing chunking settings, since existing chunks are not retroactively re-split.

The summary flags anything that will silently degrade retrieval:

```
ingested 42 manual(s), 855 chunks in 75.0s using text-embedding-3-large
  dropped 67 sub-minimum fragments (deck boilerplate: 'THE END', title slides)
  NEEDS CONVERSION (3): legacy binary Office files ...
  NEEDS OCR (1): no extractable text, likely image-only ...
```

Confirm what landed:

```bash
.venv/bin/python -m app.cli corpus
```

## 4. Score a script

**Web UI** — the primary interface:

```bash
make serve            # http://localhost:8080
```

**CLI:**

```bash
.venv/bin/python -m app.cli score \
  --brief samples/brief.md \
  --script samples/script_mixed.md \
  --product "Tea Tree Pore Ampoule"
```

**REST:**

```bash
curl -s localhost:8080/score -H 'Content-Type: application/json' -d '{
  "brief": "...", "script": "...", "product_hint": "Tea Tree Pore Ampoule"
}' | jq .
```

| Endpoint | Purpose |
|---|---|
| `POST /score` | Score a script |
| `GET /health` | Provider, models, corpus size, readiness |
| `GET /corpus` | What the validator actually knows — the honest answer to "why was my claim unverifiable" |
| `GET /runs`, `/runs/{id}` | Audit trail |
| `GET /metrics` | Score drift, retrieval health, latency, cost per run |
| `POST /eval/retrieval` | Full golden-set eval, independent of any one run |

The build context excludes the 256MB manual corpus (`.dockerignore`) — the
manuals are an *input* to ingestion, not a runtime dependency. Build context is
~13MB.

**MCP** (optional, `pip install "mcp>=1.2"`) — exposes `validate_script`,
`list_products`, `retrieval_eval`, `service_health` over stdio:

```json
{"mcpServers": {"tfs-validator": {
  "command": "/abs/path/.venv/bin/python",
  "args": ["-m", "app.mcp_server"], "cwd": "/abs/path/script-validator"}}}
```

## 5. Deploy to Cloud Run

```bash
gcloud config set project YOUR_PROJECT_ID
make ingest                  # the corpus must exist before the image is built
./deploy/cloudrun.sh
```

Builds remotely via Cloud Build, so **local Docker is not required**. The script
enables the APIs, creates the runs bucket and Firestore database, stores the API
key in Secret Manager, grants the runtime service account access, deploys, and
smoke-tests `/health` before reporting success.

One image serves the API and the frontend. Locally:

```bash
docker compose run --rm ingest     # mirrors a Cloud Run Job
docker compose up api              # mirrors a Cloud Run Service
```

### How state is handled on Cloud Run

Cloud Run instances are ephemeral and concurrent, which breaks two assumptions
the local build makes.

**The vector store is baked into the image.** It is read-only at query time --
ingestion is a separate batch job -- so the built image carries the embedded
corpus (~12MB) rather than reading it from a bucket or a database. Cold starts
are instant, there is nothing to provision for retrieval, and the corpus version
is pinned to the image version. Updating manuals means re-ingesting and
redeploying, which is an accurate description of what changed: the service's
knowledge is part of the artifact, not config it picks up later. The Docker build
fails if the store is missing, because a service with an empty corpus deploys
perfectly and marks every claim unverifiable.

**Run storage moves to GCS + Firestore.** Full run artifacts go to
`gs://<bucket>/runs/<run_id>.json`, the queryable index to Firestore, so `/runs`
and `/metrics` work across instances. Both scale to zero, preserving the cost
profile. `RUNS_BACKEND=local` keeps SQLite + disk for development; the scoring
pipeline is identical either way.

**The API key is a mounted secret**, never a plain env var -- env vars are
readable by anyone with `run.services.get` on the project.

---

## Architecture

**Ingestion** (offline, never in the request path):

```
data/manuals/*.pptx ──► Loader ──► Chunker ──► Embedder ──► SQLite + vectors
                        slide→      section-    OpenAI /
                        blocks      aware       Gemini
```

**Scoring** (per submission — the three branches run concurrently):

```
brief + script
      ├─► Brief Alignment scorer ──────────────┐
      ├─► Message Quality scorer ──────────────┤
      └─► Claim Extractor                      │
              └─► Retriever (per claim)        │
                     └─► Verifier ─► Penalty ──┴─► Aggregator ─► scores + feedback
                                                        └─► Retrieval Eval
```

### Design decisions worth knowing

**Per-claim retrieval, not per-script.** A single embedding of a whole script is
dominated by narrative and tone, so the chunk that adjudicates "contains 5%
niacinamide" routinely falls outside top-k. This is the biggest single lever on
claim-validity accuracy.

**Hybrid retrieval.** Dense cosine fused with BM25 via reciprocal rank fusion.
Dense vectors are good at meaning and bad at exact tokens — ask for "2% salicylic
acid" and pure vector search happily returns a chunk saying **5%**. RRF combines
two incommensurable score scales by rank instead of requiring a tuned weight.

**SQLite + numpy, not pgvector or an ANN index.** At ~850 chunks a brute-force
float32 dot product is sub-millisecond and *exact*; an index would trade recall
for ops burden and gain nothing. The interface is the pgvector interface, so the
backend swaps cleanly when the corpus outgrows this.

**Claim validity is arithmetic, not an LLM opinion.** Start at 10, subtract by
verdict × risk. An LLM asked for a holistic 1–10 is not reproducible run to run;
a penalty table can be shown to a brand team and argued with.

| Verdict | high risk | medium | low |
|---|---|---|---|
| contradicted | −5.0 | −3.0 | −1.5 |
| unverifiable | −4.0 | −0.9 | −0.4 |
| partially supported | −1.2 | −0.7 | −0.35 |

**Unsubstantiated high-risk claims block publication.** The manuals are silent on
almost every false medical claim ever written — treating silence as a minor
documentation gap is exactly how "cures acne" ships. A `contradicted` claim or a
high-risk `unverifiable` claim forces `needs_revision_blocking` regardless of the
weighted average.

**Mandatories are settled in code where they are literal.** A brief's
mandatories split in two: hashtags, product names and required phrases are
settled by string search; "show the texture on camera" or "end with a clear CTA"
need interpretation. An LLM call classifies each, code verifies the literal ones,
and the scorer is handed verified facts rather than asked to look. This was not
theoretical -- `gpt-4o-mini` reported "did not say the full product name" for a
script whose opening line was "I want to talk about the TFS Tea Tree Pore
Ampoule", costing 5 points. The scorer is also prevented in code from listing a
mandatory the string check proved present, not just instructed not to.

**No claims ⇒ claim axis is dropped, not scored 10.** Its weight is redistributed.
Scoring 10 would reward avoiding claims; scoring 0 would punish emotional copy for
being emotional.

**Competitor rows are guarded against.** These training decks contain competitive
pricing tables listing The Ordinary, Cosrx and Anua alongside TFS products. The
verifier is instructed that a rival's spec row never supports a TFS claim.

### Retrieval evaluation

Two different questions, reported separately on every run:

- **Golden set** (`data/eval/golden.yaml`, 20 labelled cases) — "is retrieval
  healthy?" A stable benchmark, comparable across runs, providers and chunking
  changes. Relevance is defined by *source file + keywords*, never chunk ids —
  ids change whenever chunking changes, which is when the benchmark most needs to
  stay valid. A sample runs per request; `POST /eval/retrieval` runs all of it.
- **Grounding rate** — "did *this* script's claims find evidence?" No ground truth
  exists for a script submitted five seconds ago, so this is a proxy.

Reporting only the first hides a script that hit a hole in the catalogue; only the
second lets a degraded retriever look fine on a vague script.

**Measured on the 20-case golden set over the real TFS corpus:**

| Provider / embedding model | dim | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|---|
| OpenAI `text-embedding-3-large` | 3072 | **0.95** | **1.00** | 1.00 | **0.967** |
| Gemini `gemini-embedding-001` | 3072 | 0.85 | 0.95 | 1.00 | 0.913 |

Both converge at k=5, so with the default `TOP_K=5` either is viable and Gemini
is free. Both miss the same case at k=1 (`g13`, a claim with no product hint),
which makes it an inherently hard query rather than a provider weakness.

### Logging and storage

Every run writes structured JSON logs carrying a `run_id` (Cloud Logging parses
`severity` natively), a full artifact to `data/runs/<run_id>.json` including every
retrieved chunk, and a queryable row in SQLite for trend analysis. Each run records
the provider, both model names, the prompt version and per-stage latency and token
counts — so a score that moves can be traced to a specific prompt or model change
rather than guessed at.

---

## Known limitations

- **`TFS Brochure.pdf` is image-only** and contributes nothing to retrieval. An
  OCR step in the loader would fix it.
- **3 legacy `.ppt` files** need LibreOffice conversion before ingestion.
- **Only 144/855 chunks carry a section heading** — these decks put text in
  generic textboxes rather than title placeholders, so most citations read
  `Product > p.N` rather than `Product > Section > p.N`.
- **Claim penalties are additive and unbounded**, so the claim score is mildly
  sensitive to script length. Fine at 5–10 claims; a very long, mostly-accurate
  script could score below a vague one. Fix would be to keep high-risk penalties
  absolute and make medium/low proportional to claim count.
- **It scores text, so camera-dependent mandatories cannot be verified.** A
  brief asking to "show the texture on camera" is marked `[JUDGEMENT]` and the
  scorer guesses from the script's wording. Nothing in a script can settle it.
- **It is a reviewer's assistant, not an approver.** It checks what is checkable
  against the manuals; it does not know regional advertising law or what legal
  signed off on last quarter. Every verdict carries its source quote so a human
  can overrule it in seconds.
