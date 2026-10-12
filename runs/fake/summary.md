# agent-telephone — Phase 0 pilot (fake backend)

**Synthetic data.** Agents are a seeded `FakeTeam` and the judge is `FakeJudgeBackend`: these tables check that runner, log, scoring, endpoint evaluation and analysis fit together. They say nothing about the hypotheses (the fake has no format or topology effect by construction).

- 48 runs (software_pilot, event_pilot x 2 topologies x 3 formats x clean + noise variants), 228 messages, 5016 scored units per view, 48 endpoint evaluations.
- Views: `note` = what the agent wrote; `full` = what the receiver saw (incl. a forwarded original).

## Instrument check: scorer vs injected ground truth (note view, source + hand-offs)

| method | units | agree | unscored | accuracy |
|---|---|---|---|---|
| deterministic | 1788 | 1788 | 0 | 1.000 |
| judge | 2172 | 2101 | 0 | 0.967 |

Judge pass 0 vs pass 1 (noise_p=0.05): κ = 0.927 (observed 0.963, n = 2820).

Expected judge gaps: `FakeJudgeBackend` cannot emit `distorted`, so noise-affected judge facts come out `lost`; an accent drop inside a name also breaks the literal match of judge facts that quote it.

## Survival by distance (note view; downward, non-forward hand-offs)

| kind | distance | scored | survival | lost_rate | distorted_rate | inverted_rate |
|---|---|---|---|---|---|---|
| endpoint | 4 | 264 | 0.386 | 0.568 | 0.045 | 0.000 |
| endpoint | 5 | 792 | 0.247 | 0.641 | 0.111 | 0.000 |
| handoff | 2 | 704 | 0.906 | 0.053 | 0.027 | 0.014 |
| handoff | 3 | 352 | 0.849 | 0.088 | 0.031 | 0.031 |
| handoff | 4 | 528 | 0.809 | 0.127 | 0.038 | 0.027 |
| source | 1 | 1056 | 0.960 | 0.017 | 0.023 | 0.000 |

## Survival by information distance (note view; downward, non-forward hand-offs)

| kind | info_distance | scored | survival | lost_rate | distorted_rate | inverted_rate |
|---|---|---|---|---|---|---|
| handoff | 2 | 1056 | 0.876 | 0.077 | 0.030 | 0.017 |
| handoff | 3 | 352 | 0.849 | 0.088 | 0.031 | 0.031 |
| handoff | 4 | 176 | 0.795 | 0.131 | 0.040 | 0.034 |

## Upward hand-backs (note view; reported separately, not loss)

| topology | distance | scored | survival | lost_rate | distorted_rate | inverted_rate |
|---|---|---|---|---|---|---|
| hub_relay | 3 | 528 | 0.862 | 0.089 | 0.030 | 0.019 |

## Hand-off survival by format (note view; free vs structured only)

| handoff_format | distance | scored | survival | lost_rate | distorted_rate | inverted_rate |
|---|---|---|---|---|---|---|
| free | 2 | 352 | 0.906 | 0.054 | 0.028 | 0.011 |
| free | 3 | 176 | 0.852 | 0.080 | 0.028 | 0.040 |
| free | 4 | 264 | 0.803 | 0.133 | 0.038 | 0.027 |
| structured | 2 | 352 | 0.906 | 0.051 | 0.026 | 0.017 |
| structured | 3 | 176 | 0.847 | 0.097 | 0.034 | 0.023 |
| structured | 4 | 264 | 0.814 | 0.121 | 0.038 | 0.027 |

## Forward notes: contradiction of the attached original (note view)

| direction | distance | n | scored | contradicted | contradiction_rate | silent_rate |
|---|---|---|---|---|---|---|
| down | 2 | 352 | 352 | 11 | 0.031 | 0.068 |
| down | 3 | 176 | 176 | 10 | 0.057 | 0.108 |
| down | 4 | 264 | 264 | 18 | 0.068 | 0.148 |
| up | 3 | 176 | 176 | 8 | 0.045 | 0.097 |

## Hand-off survival by fact type (note view; downward, non-forward)

| fact_type | distance | scored | survival | lost_rate | distorted_rate | inverted_rate |
|---|---|---|---|---|---|---|
| conditional | 2 | 80 | 0.925 | 0.050 | 0.000 | 0.025 |
| conditional | 3 | 40 | 0.900 | 0.050 | 0.000 | 0.050 |
| conditional | 4 | 64 | 0.828 | 0.109 | 0.000 | 0.062 |
| low_salience | 2 | 80 | 0.800 | 0.200 | 0.000 | 0.000 |
| low_salience | 3 | 40 | 0.675 | 0.325 | 0.000 | 0.000 |
| low_salience | 4 | 64 | 0.609 | 0.391 | 0.000 | 0.000 |
| name | 2 | 128 | 0.914 | 0.000 | 0.086 | 0.000 |
| name | 3 | 64 | 0.891 | 0.000 | 0.109 | 0.000 |
| name | 4 | 96 | 0.854 | 0.000 | 0.146 | 0.000 |
| negation | 2 | 112 | 0.893 | 0.036 | 0.000 | 0.071 |
| negation | 3 | 56 | 0.804 | 0.036 | 0.000 | 0.161 |
| negation | 4 | 88 | 0.864 | 0.023 | 0.000 | 0.114 |
| number | 2 | 176 | 0.955 | 0.000 | 0.045 | 0.000 |
| number | 3 | 88 | 0.955 | 0.000 | 0.045 | 0.000 |
| number | 4 | 120 | 0.950 | 0.000 | 0.050 | 0.000 |
| priority | 2 | 64 | 0.922 | 0.078 | 0.000 | 0.000 |
| priority | 3 | 32 | 0.812 | 0.188 | 0.000 | 0.000 |
| priority | 4 | 48 | 0.750 | 0.250 | 0.000 | 0.000 |
| rationale | 2 | 64 | 0.875 | 0.125 | 0.000 | 0.000 |
| rationale | 3 | 32 | 0.750 | 0.250 | 0.000 | 0.000 |
| rationale | 4 | 48 | 0.562 | 0.438 | 0.000 | 0.000 |

## Invented claims by distance (note view)

| kind | distance | messages | invented | per_message |
|---|---|---|---|---|
| source | 1 | 48 | 0 | 0.000 |
| handoff | 2 | 48 | 4 | 0.083 |
| handoff | 3 | 48 | 10 | 0.208 |
| endpoint | 4 | 12 | 0 | 0.000 |
| handoff | 4 | 36 | 11 | 0.306 |
| endpoint | 5 | 36 | 0 | 0.000 |

## Noise propagation classes (note view)

| noise_kind | kind | distance | flagged | freeze | amplify | repair | lost | drift |
|---|---|---|---|---|---|---|---|---|
| asr_name | endpoint | 4 | 0 | 6 | 0 | 0 | 0 | 0 |
| asr_name | endpoint | 5 | 0 | 12 | 0 | 0 | 0 | 0 |
| asr_name | handoff | 2 | 0 | 18 | 0 | 0 | 0 | 0 |
| asr_name | handoff | 3 | 0 | 18 | 0 | 0 | 0 | 0 |
| asr_name | handoff | 4 | 0 | 12 | 0 | 0 | 0 | 0 |
| homophone | endpoint | 4 | 0 | 3 | 0 | 0 | 0 | 0 |
| homophone | endpoint | 5 | 0 | 3 | 0 | 0 | 0 | 6 |
| homophone | handoff | 2 | 0 | 12 | 0 | 0 | 0 | 0 |
| homophone | handoff | 3 | 0 | 12 | 0 | 0 | 0 | 0 |
| homophone | handoff | 4 | 0 | 9 | 0 | 0 | 0 | 0 |
| punct | endpoint | 4 | 0 | 0 | 0 | 0 | 3 | 0 |
| punct | endpoint | 5 | 0 | 0 | 0 | 0 | 9 | 0 |
| punct | handoff | 2 | 0 | 12 | 0 | 0 | 0 | 0 |
| punct | handoff | 3 | 0 | 12 | 0 | 0 | 0 | 0 |
| punct | handoff | 4 | 0 | 9 | 0 | 0 | 0 | 0 |

## Endpoint by scenario and noise

| scenario | noise | runs | mean_score | all_passed | errors |
|---|---|---|---|---|---|
| event_pilot | asr_cfo | 6 | 0.783 | 0 | 0 |
| event_pilot | clean | 6 | 0.833 | 0 | 0 |
| event_pilot | homophone_guests | 6 | 0.842 | 0 | 0 |
| event_pilot | punct_alcohol | 6 | 0.842 | 0 | 0 |
| software_pilot | asr_signer | 6 | 0.731 | 0 | 0 |
| software_pilot | clean | 6 | 0.796 | 0 | 0 |
| software_pilot | homophone_two | 6 | 0.750 | 0 | 0 |
| software_pilot | punct_channel | 6 | 0.722 | 0 | 0 |

## Endpoint by scenario and topology

| scenario | topology | runs | mean_score | all_passed | errors |
|---|---|---|---|---|---|
| event_pilot | hier_chain | 12 | 0.842 | 0 | 0 |
| event_pilot | hub_relay | 12 | 0.808 | 0 | 0 |
| software_pilot | hier_chain | 12 | 0.741 | 0 | 0 |
| software_pilot | hub_relay | 12 | 0.759 | 0 | 0 |

## H4: endpoint by scenario and hand-off format

| scenario | handoff_format | runs | mean_score | all_passed | errors |
|---|---|---|---|---|---|
| event_pilot | forward | 8 | 0.812 | 0 | 0 |
| event_pilot | free | 8 | 0.819 | 0 | 0 |
| event_pilot | structured | 8 | 0.844 | 0 | 0 |
| software_pilot | forward | 8 | 0.736 | 0 | 0 |
| software_pilot | free | 8 | 0.785 | 0 | 0 |
| software_pilot | structured | 8 | 0.729 | 0 | 0 |

## H7 plumbing: fact label at the last hand-off vs its endpoint check (note view)

| label_at_last_handoff | facts | endpoint_pass_rate |
|---|---|---|
| distorted | 26 | 0.000 |
| inverted | 18 | 0.056 |
| lost | 45 | 0.089 |
| preserved | 499 | 0.918 |

Excluded: forward runs, loss-insensitive facts, and facts whose check failed only because a fact it depends on failed. A preserved fact can still fail: hidden tests tagged with several facts fail for all of them, and the endpoint author can drop a fact itself.
