# Model evaluation — memory extraction and classification

Measured 2026-08-17 against `services/shared/shared/memory.py::extract_typed_memories`
and the taxonomy in `services/shared/shared/taxonomy.py`, both as shipped on
`fix/memory-taxonomy-extraction` (`f5bf1e3`). Read-only replay over production
`session_traces`. No memories were written.

---

## Recommendation

**Keep `xai/grok-4.1-fast-non-reasoning`.** It was the most stable model on the
metric that matters — **set stability 0.69** (Jaccard across runs of the same
transcript), against 0.50 for the next-best viable candidate — while also being
the cheapest (**$0.59 per 1,000 sessions**, a quarter of Gemini's $2.60) and the
only model with zero failures of any kind across 48 calls. The two candidates
that beat it on classification quality are disqualified by a constraint that has
nothing to do with quality: production's LLM timeout is **10 seconds**
(`DEFAULT_LLM_TIMEOUT_S`, unset in Railway on both services), and
`gpt-5.6-luna-pro` exceeded it on **39 of 48 calls**, `deepseek-v4-flash-0731` on
**41 of 48**. Under current configuration both would return `None` from
`call_llm` and store *nothing* on most sessions — the exact irreversible loss
this evaluation exists to prevent.

**The premise of this evaluation did not reproduce, and that is the more
important finding.** The brief describes the incumbent producing 282 and 376
items on identical inputs. Measured across the whole 16-trace corpus, three
times: **91, 92, 93 items — a 2% spread.** Whatever produced the 282/376 figures,
it was not this prompt against this corpus. The incumbent's aggregate output is
stable. What is *not* stable, and what the brief is right to worry about, is
which items come back: roughly **a third of items do not recur across runs**
(core 0.66), and the model dropped two of five known-durable reference items in
**3 out of 3 runs**. That is a prompt and pipeline problem, not a model-choice
problem, and swapping models does not fix it — the best retention any model
managed was 7 of 15.

---

## Scope corrections

Three things in the brief did not match the code. They changed what was measured.

1. **`taxonomy.py` is not on the default checkout.** It exists only on
   `fix/memory-taxonomy-extraction` (commits `d269c56`, `f5bf1e3`, both dated
   today). The evaluation was run against that branch.
2. **`extract_facts` is not the task described.** It returns a flat `list[str]`
   with no classification and no durability rule; its prompt is two sentences.
   The function that takes a transcript and returns items classified
   `decision|plan|fact|note|observation` with transient trivia declined is
   **`extract_typed_memories`**. That is what was evaluated. `extract_facts`
   survives only for the verification worker and was left alone.
3. **`VERIFICATION_MODEL` on `mcp_server` is the string `Grok 4.1 Non
   Reasoning`** — a display name, not a model id. It happens to resolve as a
   gateway alias to the same model, so production works, but it is a
   one-rename-away outage and is not the literal id set on `async-worker`.

Production call shape, replicated exactly: `extract_typed_memories(combined)`
with `combined = _combine_messages(req.messages)` (`capture.py:841`), i.e. only
`output`, no `task`, no `input_data`; transcript tail-capped at 16,000 chars;
`temperature=0.0`.

---

## Method

| | |
|---|---|
| Input | 16 production `session_traces` rows with `message_count >= 4` (of 20 total), S3 bodies fetched from R2 |
| Prompt | The shipped one, unchanged — `DURABILITY_RULE` + `TAXONOMY_RULE` + `TYPED_OUTPUT_SHAPE` imported from the branch |
| Parsing | The shipped `_extract_json` and `taxonomy.parse_typed_items`, including the `is_transient` backstop |
| Runs | 3 per model per trace = 192 calls per replication |
| Params | `temperature=0.0`, no `max_tokens` — as production |
| Replications | Two independent full passes (A and B). B is the clean one and is quoted throughout; A is shown where it disagrees |

**One deliberate deviation.** The HTTP timeout was set to 150s, not production's
10s. At 10s, DeepSeek and Luna return nothing at all and their quality is
unmeasurable. Timeout behaviour is therefore reported separately, as the
`calls > 10s` column, rather than folded into the failure counts. This is the
only parameter changed, and it favours the slow models.

**A measurement fault worth recording.** Replication A ran with two harness
processes contending on the gateway (a `nohup` job that outlived its shell).
Latency and cost from A are unusable and were discarded; B was re-run as a
single process. The *content* metrics from A are still shown because the
disagreement between A and B is itself the honest error bar.

---

## 1. Viability — failure modes, latency, cost

Replication B, 48 calls per model.

| model | errors | non-JSON | truncated | p50 | max | **calls > 10s** | $/session | **$/1,000** |
|---|---|---|---|---|---|---|---|---|
| **grok-4.1-fast-non-reasoning** | **0** | 4 | 0 | **3.1s** | **5.1s** | **0 / 48** | 0.00059 | **0.59** |
| deepseek-v4-flash-0731 | 16 | 18 | 0 | 107.4s | 150.2s | 41 / 48 | 0.00136 | 1.36 |
| gemini-3.7-flash | 0 | 0 | 0 | 6.5s | 10.4s | 1 / 48 | 0.00260 | 2.60 |
| gpt-5.6-luna-pro | 0 | 0 | 0 | 18.9s | 34.2s | 39 / 48 | 0.00506 | 5.06 |

All four models resolve and respond through `https://litellm.oppla.dev/v1`.
Costs are the gateway's own reported `usage.cost`.

- **DeepSeek is not a candidate.** 16 hard timeouts and 18 unparseable responses
  out of 48 — a third of sessions produce nothing even at a 150s budget. Its
  `$1.36/1k` is also understated, because 34 of 48 calls bought nothing.
- **Luna's `$5.06/1k` is 8.6× the incumbent** and its p50 is nearly 2× the
  production timeout.
- Gemini's single >10s call (10.4s) is a warning, not a disqualification.
- The 4 non-JSON responses from grok are all empty-output runs on short
  transcripts, which `parse_typed_items` already degrades to `[]` safely.

## 2. Stability

The deciding table. `setJacc` is the mean pairwise Jaccard between runs of the
same transcript, matching items by token-set similarity ≥ 0.5 (exact string
match would be unfairly strict — models reword). `core` is the fraction of items
present in *all three* runs. `drift` is the share of runs where ≥80% of items
carry a single kind.

| model | items/run | mean per-trace spread | **setJacc** | core (3/3) | drift |
|---|---|---|---|---|---|
| **grok-4.1-fast-non-reasoning** | 0–13 (med 6) | **1.6** | **0.69** | **0.66** | 37% |
| deepseek-v4-flash-0731 | 0–34 (med 1) | 9.4 | 0.44 | 0.43 | 33% |
| gemini-3.7-flash | 0–6 (med 3) | 0.6 | 0.50 | 0.39 | 56% |
| gpt-5.6-luna-pro | 0–19 (med 10) | 2.2 | 0.35 | 0.29 | **31%** |

Whole-corpus totals per pass — the direct test of the brief's 282/376 claim:

| model | pass 1 | pass 2 | pass 3 | spread |
|---|---|---|---|---|
| **grok** | 91 | 92 | 93 | **2%** |
| deepseek | 88 | 115 | 105 | 23% |
| gemini | 50 | 49 | 50 | 2% |
| luna | 129 | 140 | 131 | 8% |

Replication A gave the same **ordering** but different magnitudes (grok 0.52
setJacc / 0.49 core; gemini 0.40 / 0.28; deepseek 0.31 / 0.40; luna 0.31 / 0.21).
Grok ranked first on set stability in both passes. Treat the absolute values as
±0.15; treat the ordering as the result.

**What this means.** Item counts are stable; item *identity* is not. At core
0.66, roughly one item in three that a run produces will not be produced by the
next run. Because extraction is single-shot and irreversible, that is a real
one-in-three chance of losing any given durable item — and no candidate model
improves on it. Grok is simply the least bad.

## 3. Classification quality

Every unique `decision` and `plan` item was graded by `anthropic/claude-opus-4.8`
— not a candidate, so nothing grades itself — against the shipped
`TAXONOMY_RULE` and `DURABILITY_RULE`. `label_ok` = a reader would agree with the
assigned kind; `durable` = the item passes the durability test at all.

| model | items graded | label_ok | durable |
|---|---|---|---|
| grok-4.1-fast-non-reasoning | 51 | 92% | 98% |
| deepseek-v4-flash-0731 | 88 | 88% | 98% |
| gemini-3.7-flash | 50 | 88% | 100% |
| gpt-5.6-luna-pro | 168 | **95%** | 99% |

**Do not read these as separating the models.** The same grader on replication A
gave grok 87%, deepseek 95%, gemini 94%, luna 98% — the ordering inverted for
deepseek and gemini between passes. The judge is as noisy as the thing it judges.
The only claim these numbers support is that **all four label roughly 9 in 10
items in a way a reader would accept**, and that differences under ~10 points are
not resolvable at this sample size. Luna is plausibly best; it is not
demonstrably best.

Register drift is real but not catastrophic, and it is not what the brief
described. No whole transcript came back as all-plans. Single-register runs
(≥85% one kind, ≥5 items) occurred in 6/35 eligible grok runs, 0/21 deepseek,
2/14 gemini, 3/38 luna — and the register is nearly always `fact`, which for
these transcripts is often correct. Two transcripts (`014d10c7`, `2825e0c1`)
account for most of it across every model, which points at the transcripts rather
than the models.

## 4. Junk rejection

Two measurements, reported together with retention (§5) as required.

- **The deterministic backstop is inert.** `is_transient` rejected **0.0–0.7%**
  of model-emitted items (grok 0/276, deepseek 0/308, gemini 1/150, luna 0/400).
  The prompt is doing all the work; the regex layer catches essentially nothing.
  That is the designed outcome, but it also means there is no safety net if a
  model regresses.
- **An independent regex screen over *kept* items** flags 2.3–5.8% as
  junk-shaped: grok 13/276 (4.7%), deepseek 7/308 (2.3%), gemini 5/149 (3.4%),
  luna 23/400 (5.8%). This screen over-fires — several hits are genuinely
  durable — so read it alongside the judge's `durable` rate of 98–100%.

Real junk does get through. Quoted verbatim, all from grok:

> `Commit 7fd152a on branch fix/migration-039-role-assertions (off main) updates three test_migration_039 assertions from `member` to `write`…` — classified **decision**

That is a commit SHA and a branch name, which `DURABILITY_RULE` names explicitly
as point-in-time VCS state to drop, stored as a settled architectural choice.

> `New tests added in agent-plan.test.ts covering required unrecognized file uploads on both heuristic and LLM paths, mutation-verified.` — classified **decision**

A test-run outcome, also named in the rule.

> `Dashboard currently covers Vex reliability product (executions, sessions, agents, alerts, guardrails)…` — "currently" is doing load-bearing work in a memory meant to survive a week.

Note also that **`observation` is never used** — 1 item out of 1,133 kept items
in replication B, and 0 out of 1,337 in replication A. The five-way taxonomy is
in practice a four-way one.

## 5. Durable-knowledge retention

Against the five reference items from the review. Each appears in exactly one
trace; all five were confirmed present in the input text before scoring. Score
is hits out of 3 runs.

| reference item | trace | grok | deepseek | gemini | luna |
|---|---|---|---|---|---|
| minSdk 23 / flutter_webrtc | `582c3866` | 1/3 | 1/3 | **3/3** | 2/3 |
| i18n translations cached at server start | `1b7d55d7` | **3/3** | 1/3 | 0/3 | 2/3 |
| Railway auto-deploys on merge to main | `a57a2860` | **0/3** | 1/3 | 0/3 | 0/3 |
| `hasPositiveFillEvidence` only tightens `submitReady` | `31dba0f3` | **0/3** | 0/3 | 0/3 | 0/3 |
| M1 excludes progress timeline / re-scans | `014d10c7` | **3/3** | 0/3 | 0/3 | **3/3** |
| **total (of 15)** | | **7** | 3 | 3 | **7** |

**This is the worst result in the report and it applies to every model.** Fewer
than half of known-durable items survive. `hasPositiveFillEvidence` — an
invariant stated plainly in the transcript — was kept by **no model, in no run,
ever**. "Railway auto-deploys on merge to main" was kept once in twelve
model-runs.

Read against §4: the models are not being over-conservative in a way that trades
junk for recall. They keep 2–6% junk *and* drop half the durable material. Both
error modes at once.

Replication A gave grok 6, deepseek 9, gemini 3, luna 8 — deepseek moved 9→3.
**With 5 items × 3 runs = 15 observations per model, this table cannot rank
models.** It is evidence that retention is broadly poor, and nothing more.

## 6. Two-call split — extract, then classify

Tested on the incumbent, 16 traces × 3 runs, same rule text, split into an
extraction call (durability only) and a classification call (taxonomy only).

| | med items | spread | setJacc | core | drift | $/1k | p50 |
|---|---|---|---|---|---|---|---|
| single-call (shipped) | 6 | 1.6 | **0.69** | **0.66** | **37%** | **0.59** | **3.1s** |
| two-call split | 8 | 1.8 | 0.56 | 0.46 | 50% | 0.82 | 5.6s |

Type mix, single → split: decision 22%→22%, plan 7%→7%, fact 60%→56%,
note 11%→14%.

**The split does not help — it hurts.** The type mix is essentially unchanged,
so the hypothesis that the model picks a register per transcript rather than per
item is not supported. Meanwhile set stability falls (0.69→0.56), the fraction of
items surviving all three runs falls (0.66→0.46), drift *rises* (37%→50%), cost
rises 1.4× and latency 1.8×. Separating the two questions gives the extraction
call a second opportunity to be nondeterministic. **Do not adopt it.**

---

## Where the sample is too small to conclude

Stated plainly, because a confident ranking here would be built on noise.

- **16 traces is not many**, and they are not independent: several are long
  sessions from the same few projects, and 10 of 16 hit the 16,000-char tail cap,
  so the model sees only the end of each. Two transcripts drive most of the
  drift signal.
- **Retention rests on 5 items × 3 runs per model.** One item flipping moves a
  model two places. The 7 / 3 / 3 / 7 column should not be used to rank.
- **Classification quality differences under ~10 points are not real** — the
  same judge on the same task reordered two models between replications.
- **The judge is a single model.** Its `label_ok = false` calls include arguable
  ones (it insists author-attribution rules are `note`, not `decision`; that is a
  defensible reading but not the only one).
- **Latency and cost are one-shot**, taken on one afternoon through one gateway.
  Gemini's 10.4s max and grok's 5.1s max are close enough to the 10s timeout that
  a slow day changes the viability verdict for Gemini, though not for grok.
- **`session_traces` holds ~3 days of data.** Nothing here says how these models
  behave on transcript shapes the corpus does not currently contain.

## What was not tested

- **`extract_facts` on the verification path** — out of scope; it has no
  taxonomy or durability rule and a different consumer.
- **The 10s production timeout as a live variable.** Behaviour under the real
  timeout is inferred from measured latency, not observed. Confirming it would
  require a config change, which was out of scope.
- **Anything downstream of extraction** — `write_scoped_memory`, dedup, the
  supersedes graph. Items counted here as "kept" are pre-dedup.
- **Redaction.** S3 bodies are already redacted by `build_captured_payload`, so
  `redact()` was not re-applied; egress behaviour was not exercised.
- **Prompt variants other than the two-call split.** If the goal is fixing
  retention rather than picking a model, that is where the next experiment
  belongs — see below.

## What to do instead of changing the model

The measurement does not support a model swap. It does support these, in order:

1. **Retention is the defect.** No model keeps more than half the known-durable
   items, and one reference invariant was kept by nobody. This is worth an
   experiment on the prompt — the durability rule spends ~1,260 characters
   listing what to *drop* and two sentences on what to *keep*.
2. **Set stability is a third short of deterministic at `temperature=0.0`.**
   Since extraction is single-shot and irreversible, the cheapest available fix
   is *n*-of-3 sampling with a union merge: at $0.59/1,000, three grok passes
   cost $1.77/1,000 — still a third of Luna's single pass — and would convert a
   0.66 core into near-complete coverage. This is the highest-value change in
   the report and it does not require changing models.
3. **The backstop catches nothing (0.0–0.7%).** Either it is redundant and
   should be acknowledged as such, or it is too narrow to be the safety net its
   docstring claims.
4. **`observation` is dead** — 1 item in 1,134. Either drop it from the taxonomy
   or stop offering it as a last resort.

## Reproducing

Harness, trace cache, raw responses and judge verdicts are in the session
scratchpad (not committed — the trace bodies are production transcripts):
`runner.py` (replay), `analyze.py` (metrics), `judge.py` (grading),
`split.py` (two-call experiment), `fetch_traces.py` (read-only trace pull).
Credentials are read from `railway variables --service async-worker --kv` at run
time and never written to the repo.
