# agent-telephone — design

Does a request survive being passed through a team of agents? When a brief travels
user → lead → role agent → role agent, which parts arrive intact, which are lost,
which are distorted, which are invented, and does that change the final outcome?

Status: **pre-registered design, Phase 0 (no real model calls)**. Hypotheses below are
fixed before any real run; changes after data is seen go to the changelog at the end.

## 1. Realism constraints (from the author)

- In flat peer graphs a message rarely travels more than 1 edge (sometimes 2–3 when a
  session forwards to the one that actually owns the topic). Long chains are unrealistic.
- The natural place for the effect is a **hierarchy**: a lead that talks to the user and
  distributes work to role agents (PM, architect, implementer, tester, QA…), i.e. a human
  team turned into an agent team.
- Maximum realistic distance from the original message: **3–4 edges**.
- Where a realistic setup exists we use it; where not, a small artificial setup is
  acceptable to check whether the effect exists at all.
- Speech-to-text is **not a hop**: it is noise applied to the original message.

## 2. Domains

| Domain | Team | Endpoint (functional consequence) |
|---|---|---|
| `software` | client → lead → PM → architect → implementer (+ QA) | final implementation run against **hidden tests** derived from the original brief |
| `event` | client → account manager → logistics coordinator → vendor liaison (catering / venue / transport) | final **plan as JSON** (schema-fixed) checked against the original facts |

Each scenario is a brief with **15–25 atomic facts**, each tagged with a type:

| Type | Example |
|---|---|
| `name` | rare proper nouns: "Siobhán Ní Dhúill", "Wojciechowski", a field called `kyc_tier_b` |
| `number` | numbers, units, dates, amounts: "max 3 retries", "142 guests", "17:45" |
| `negation` | "do NOT log the email", "no shellfish, not even in stock" |
| `conditional` | "only if the user is in the EU", "unless it rains, then hall B" |
| `priority` | "the export matters more than the dashboard" |
| `rationale` | the *why* behind a decision ("because legal flagged it in March") |
| `low_salience` | asides: "oh, and the CEO is left-handed — seat accordingly" |

Every fact has a deterministic check where possible (`regex`/exact string for names and
numbers) and an LLM-judge check otherwise.

## 3. Topologies

| Id | Graph | Max distance | Runner |
|---|---|---|---|
| `hier_chain` | user → lead → r1 → r2 → r3 (nested delegation) | 4 | Claude Code nested subagents; API harness |
| `hub_relay` | user → lead → r1 → **lead** → r2 (lead rewrites r1's output for r2) | 4 | Claude Code; API harness |
| `peer_route` | user → A → B → C where B is not the owner and forwards to C (no hierarchy) | 3 | cross-session messaging (small, artificial arm); API harness |

Every message is recorded as one row of the **message log** (JSONL):

```json
{"run_id": "...", "scenario": "...", "condition": {...}, "edge": ["lead", "pm"],
 "distance": 2, "sender": "lead", "receiver": "pm", "text": "...",
 "model": "...", "usage": {"input_tokens": 0, "output_tokens": 0}}
```

`distance` = number of edges from the original message (user → lead is distance 1).
Each row also records `direction` (`down` = delegation towards the endpoint, `up` = hand-back
to a caller) and `info_distance` = 1 + the smallest distance at which the writer had received
source-derived content. They differ in `hub_relay`: the lead still holds the brief when it
writes to r2, so lead → r2 is at distance 4 but information distance 2, and r1 → lead (d3) is
an upward hand-back.

## 4. Factors

| Factor | Levels | Notes |
|---|---|---|
| Distance | 1–4 (measured on every message of every run) | not a manipulated factor; read off the log |
| Topology | `hier_chain`, `hub_relay` (+ `peer_route` small arm) | |
| Model | Claude Haiku 5.5, Claude Sonnet 5.5 (subscription, Claude Code); gpt-6-luna via gateway (API harness) | homogeneous chains first; mixed chains only after |
| Handoff format | `free` (prose) / `structured` (requirement list with IDs) / `forward` (original attached + own note) | the practical intervention |
| Source noise | `clean` / `asr` | see §5 |
| Domain | `software`, `event` | |

Order of work (close before widening): pilot one cell, validate instruments, then grid.

## 5. Noise

Applied only to the original message, never to inter-agent messages.

- `asr_name`: proper noun replaced by a phonetically plausible mis-transcription
  ("Siobhán" → "Chevonne").
- `homophone`: word-level homophone/near-homophone errors.
- `punct`: **minimal pairs** where punctuation changes meaning: list commas, scope of a
  negation ("No, do it this way" vs "No do it this way"), sentence boundaries
  ("Don't stop. Deploying is fine." vs "Don't stop deploying. Is fine.").

Realistic generation (optional, Phase 1+): TTS of the clean brief → Whisper → diff;
otherwise synthetic, marked as such. Each noise variant records `affected_fact_ids`, a prose
`wrong_reading`, a machine-checkable `wrong_pattern` (regex on number-normalised text), the
`wrong_context` where that reading legitimately sits, and whether it is `repairable` (an
ASR-mangled name is not: nothing in the brief allows restoring it). Per (noisy run, affected
fact, message), on the agent-written note only, classes are:
**flagged** (wrong reading present with a doubt marker: "please confirm", "typo", "?"),
**amplify** (wrong reading in a sentence outside its context: it spread to another fact;
repeating it in context is not amplification), **freeze** (present, in context, unquestioned),
**repair** (absent, fact preserved), **lost**, **drift** (absent, fact distorted otherwise).
Labels and counts always come from the note view.

Agent-generated noise is measured without injecting anything: in `clean` runs, any change of
a `name`/`number` fact is noise produced by the agents themselves.

## 6. Outcomes

Per fact × message (the unit of analysis):

| Label | Meaning |
|---|---|
| `preserved` | present with the same meaning |
| `lost` | absent |
| `distorted` | present with a different meaning (incl. a wrong name/number) |
| `inverted` | polarity flipped (negations, conditionals) — special case of distorted |

Per message: `invented` = claims not traceable to the brief that a downstream agent would
act on (counted, with text spans).

Per run: endpoint score (hidden tests passed / plan checks passed).

`forward` hand-offs: the agent writes a note and the original is attached verbatim, and the
note is told not to restate the brief. For them the per-fact unit is **contradiction** — did
the note contradict the attachment (`distorted`/`inverted`)? A silent note (`lost` in the note
view) is not a loss, and the full view is degenerate (the attachment always carries the fact).

Endpoint checks: each hidden test / plan check asserts only on the facts it is tagged with
(verified by breaking each fact in turn); facts every check needs (`universal`, e.g. the entry
point) and explicit dependencies (`depends`, e.g. "CEO at the left end" needs the CEO seated)
are declared; facts whose check passes by default when lost (`loss_insensitive`, e.g. "never
both channels") are declared and excluded from H7.

Scoring stack:
1. deterministic checks first (exact / normalised / fuzzy match for names and numbers;
   fuzzy-but-not-exact ⇒ `distorted`). Numbers, times, dates and money are compared on
   canonical text (spelled-out numbers → digits, times → 24h HH:MM, dates with a year → ISO,
   €18.5k → €18500) and through context-anchored slots, so a wrong value is `distorted`, not
   `lost`, and list/table renderings match like prose. Facts whose polarity or behaviour an
   anchor cannot see (which opt-out field wins, what happens inside the quiet window, where the
   reception is) are also sent to the judge, which can override a preserved anchor;
2. LLM judge for the rest, **cross-family** with respect to the agents being judged,
   two passes, agreement reported as Cohen's κ; a manually labelled subset is the anchor;
3. judge outputs are schema-validated JSON; invalid → retried once, then marked `unscored`.

## 7. Hypotheses (pre-registered)

- **H1** Fact survival decreases with distance, with most of the loss in the first
  hops where compression happens (loss at distance 1–2 > loss at 3–4, per hop).
  Tested on **downward, non-forward** hand-offs only, stratified by format; upward hand-backs
  are reported separately; information distance is reported alongside nominal distance.
- **H2** Survival depends on type: `low_salience` and `rationale` are lost most; `negation`
  and `conditional` are the most often *inverted/distorted*; `name` facts are distorted
  rather than lost.
- **H3** A noisy name in the source is **frozen** at distance 1 (treated as fact) and not
  questioned downstream; punctuation minimal pairs propagate the wrong reading.
- **H4** Handoff format: `forward` > `structured` > `free` on the **endpoint score** (and on
  the endpoint-producing role's artifact), not on hand-off survival: the attached original
  makes `forward` survive by construction in the full view and "lose" by construction in the
  note view. Hand-off survival is compared only between `free` and `structured`.
- **H5** `hub_relay` loses more than `hier_chain` at equal distance (the lead rewrites twice).
  Tested on downward, non-forward hand-offs at equal **information distance**, with nominal
  distance reported too.
- **H6** Smaller models lose more; the difference is concentrated in H2's fragile types.
- **H7** Fact loss predicts the endpoint: runs with a lost/inverted fact fail the
  corresponding hidden test. Note view of the last downward hand-off; `forward` runs excluded
  (the endpoint author read the attachment); `loss_insensitive` facts excluded; facts whose
  check failed only through a declared dependency (`universal` / `depends`) excluded.

## 8. Runners and budget

- **Phase 0 — fake LLM, zero cost.** A deterministic `FakeBackend` returns scripted handoffs
  with known injected distortions (drop / swap name / flip negation / invent). The whole
  pipeline (runner → log → scoring → analysis) is tested against ground truth it knows.
- **Phase 1 — Claude Code with the subscription.** `claude -p` with `--agents` (one
  definition per role, own model, `tools` restricted so the delegation path is forced),
  `--output-format stream-json --verbose --forward-subagent-text`; the log is rebuilt from
  `parent_tool_use_id`. Must run **isolated** from the author's config (no user CLAUDE.md,
  hooks, memory, plugins, language settings) in a throwaway working directory.
- **Phase 2 — API harness through the LiteLLM gateway** (gpt-6-luna; OpenAI-compatible).
- **Phase 3 (optional) — Anthropic API credits.** Hard cap USD 500, **no overspend allowed**.
  The `BudgetGuard` refuses any call whose worst case (prompt tokens + `max_tokens` at list
  price) would push cumulative spend over the guard limit (USD 400 by default, never above
  USD 450); spend is persisted to one absolute ledger outside the repo, survives restarts and
  fails closed (a missing ledger is refused unless explicitly initialised). A call that fails
  after it may have been billed, or reports no usage, is charged its reserved worst case; an
  actual cost above the reservation halts the ledger until a human clears it. SDK retries are
  off; every attempt takes its own reservation. Batches API + prompt caching where possible
  (cached calls are refused until cache prices are configured).
  Prices live in a config file and must be filled from the official price list before use.
  Every paid batch is estimated (time + cost per provider) and approved before launch.

## 9. Known pitfalls (from the capture in `fixtures/cc_nested_stream.jsonl`)

- Without isolation, the session loaded the author's hooks and global CLAUDE.md and replied
  in Spanish — the config contaminates the agents.
- A PM given the `Agent` tool delegated to `general-purpose` instead of `implementer`, which
  then wrote real files into the repo. Role agents must have tools restricted
  (`Agent(implementer)`-style allow-listing or `disallowedTools`) and run in a temp dir.
- Subagents ran in the background and the lead received notifications: the message order in
  the stream is not the causal order; reconstruct edges from `parent_tool_use_id` and
  `tool_use` / `tool_result` pairs, not from line order.

## Changelog

- 2026-10-12 — initial design.
- 2026-10-12 — pre-data review fixes (no real runs yet): H4 moved to the endpoint score;
  H1/H5 restricted to downward non-forward hand-offs, with `direction` and `info_distance`
  logged per message; `forward` scored as contradiction; H7 exclusions (forward runs,
  loss-insensitive facts, declared dependencies); hidden tests rewritten to assert only on
  their tagged facts; Coachline made the vendor's default provider and a logging house rule
  given to the implementer so those negations are observable at the endpoint; number/time
  checks on canonical text with slots and all-match semantics; polarity/behaviour facts also
  judged; machine-checkable noise patterns with `flagged`/context-based `amplify`;
  fail-closed budget ledger; sandboxed endpoint execution.
- 2026-10-12 — **post-data instrument fixes** (after the first 45 Claude Code pilot runs; a hand audit
  of 198 labels found ~15% systematic errors, so the deterministic H1/H7 numbers before this entry are
  void). Scoring: the exact pattern now wins over slots (a slot conflict counts only in the sentence
  that carries the match, and not when the expected value is also in that sentence); `f_first` /
  `f_second` are anchored on the 48 h / 2 h offset instead of the ordinal (the 48 of "the 48-hour
  point" and of the priority sentence excluded); in a noisy run the variant's `wrong_pattern` on an
  affected fact is `distorted` (before the accent/fuzzy fallback), so ASR names are no longer `lost`;
  code/JSON is canonicalised (`\uXXXX` escapes, quoted keys, `timedelta(hours=N)`, `time(H, M)`,
  `X_HOUR = H`, `hour >= H`); event slots are underscore-tolerant (`dinner_start`), need a connector
  (at/from/start/:) next to the time, the date slot needs a date/event anchor, and the guest slot no
  longer skips numbers. Noise classes: doubt must be about the name in the hit's own sentence (a
  "confirm" about the venue is not `flagged`); code lines and fenced blocks are not prose (a reading
  only there is `freeze`); the ASR contexts accept `SIG_` / "ends with" / seating. H7: the
  noise-affected facts of noisy runs are excluded from the main table and reported per fact, flagged;
  a forward note that repeats its attachment's noise reading is not a contradiction (`noise_reading`).
  Endpoint checks (no runs yet for scenarios 2/3): every event budget check asserts only the cap (an
  honest "no quotes yet" total of 0/null passed nothing before); `total <= cap` is an untagged
  consistency check outside the score; event_3's coffee/panel checks accept "foyer of the Aula" and
  "all sessions except the panel", and its overflow threshold is 260 (> 238 registered, so not booking
  is right under every reading, homophone included); software_2's hidden tests put NOW and period_end
  at the same time of day (any day-count convention gives 15/10 days); the punct wrong patterns of
  event_2/event_3 no longer count correct paraphrases ("when more than 25", "not to be streamed").
  Real spans from the audit are regression fixtures in `tests/test_real_spans.py`.
- 2026-10-12 — **Endpoint policy (decided after seeing pilot data):** implementers sometimes bundle their
  own tests into the artifact; when those import the module under an imagined file name, the whole
  artifact fails to import and every hidden test fails (3 software pilot runs scored 0/18 for that
  reason alone). `endpoint.strip_self_tests` now removes, at top level, `test_*` functions, `Test*` /
  `TestCase` classes, pytest-decorated helpers, `__main__` blocks and imports that cannot resolve,
  before the hidden tests run. Rationale: the hidden tests measure whether the brief survived, not the
  packaging of the agent's own tests. Each endpoint row records `self_tests_stripped`; results are
  reported with and without this policy. Software pilot endpoint means moved from 0.889/0.806
  (haiku/sonnet) to 0.975/0.972.
- 2026-10-12 — **No human labels.** The blind 150-item sheet is labelled by an independent model
  annotator and disagreements with the judge are adjudicated by a third model; agreement is reported
  as model-anchored, not human-anchored.

- 2026-10-13 — **API harness: senders are told who receives their message** (before any API-arm
  data). In the Luna smoke test the PM wrote the architect a request to write requirements instead
  of the requirements, the architect forwarded it, and the implementer refused. Claude Code agents
  always know the receiver (the Agent call names the subagent); the API harness did not. Every
  hand-off instruction now names the receiver's role (or "back to the … who delegated" for upward
  hand-backs). Full grid: all four noise levels, 3 replicates (`-s1`, `-s2` suffixes), fresh run
  directories separate from the pilot used to fix the instrument.
- 2026-10-14 — **Judge retest on a 20 % sample for the Claude grid** (before any retest data of that
  grid). Test-retest kappa (0.908 on the pilot) only needs a sample; the full second note pass
  (4,084 calls) does not fit the ChatGPT usage limit (429 after ~3,700 calls). Runs are sampled
  whole by a stable hash of the run id (`--retest-fraction 0.2`). The Luna grid's retest (Claude
  judge) was already running in full and stays full.
- 2026-10-14 — **Post-data fix: negated vendor mentions** (`e_no_coachline`, `w_no_lumiere`). The
  substring check failed any plan naming the excluded vendor, including "TBC (not Coachline)" and
  "Non-Coachline provider", so both facts scored 0 % in every model. A mention preceded by a negation
  is now an honoured exclusion: the field holds one provider, so the vendor counts as booked only if
  it is named and the field carries no negation/exclusion marker (`books()`; a first version that
  only looked before the name still missed "Lumière Fotografia excluded", "must NOT be booked" and
  Spanish phrasings). Across both grids 3 plans genuinely book the excluded vendor; regression tests. Endpoint numbers computed before
  2026-10-14 for these two facts are void; the judge labels are unaffected.
- 2026-10-14 — **Isolation gap found after the runs: the account identity.** Claude Code adds the
  logged-in account's name and email to the main session's context even with `--system-prompt`
  replaced and `--setting-sources ""`. In 17 of 853 Claude grid runs the lead used them (treating
  the email as the client's, or naming the account holder as the go-between). No scored fact
  depends on it; the published copy of the data replaces the name and email.
