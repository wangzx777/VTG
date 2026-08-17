# VTG research profile

## Research objective

Prioritize literature for Video Temporal Grounding / Video Temporal Retrieval,
with special attention to large multimodal or video language models. The triage
decision estimates reading value for this research program; it is not a venue
quality score or peer-review verdict.

## Core questions

- Does the paper predict, retrieve, or reason about temporal boundaries or
  intervals in video from language or another query?
- Does it represent the current Video-LLM / multimodal LLM direction?
- Does it establish a major alternative family such as DETR-style grounding,
  proposal/ranking, boundary modeling, or a specialized grounding model?
- Does it materially change training, post-training, reward design, temporal
  representation, long-video handling, or generalization?
- Is it a key baseline, dataset, metric, evaluation protocol, or predecessor that
  later core work depends on?

## Advisor-designated anchors

Treat UniTime, Time-R1, and TimeLens2 as `Core` unless identity has been
misresolved. Verify the exact paper; do not rely on name matching alone.

## Preferred vocabulary

Use the paper's own task name when precise. Normalize close VTG names only when
the paper actually performs temporal localization/retrieval. Do not label generic
video QA, dense captioning, moment classification, or temporal reasoning as VTG
without a boundary/retrieval output.

Typical method families include `Video-LLM`, `DETR-based`,
`Transformer-based`, and `Specialized Model`. Use a different concise family if
those labels would conceal the defining paradigm.

Typical focus labels include `RL/Post-training`, `Boundary Modeling`,
`Long Video`, `Generalization`, `Temporal Representation`, `Data/Benchmark`, and
`Evaluation`. Separate genuinely co-primary focus terms with `;`; avoid keyword
dumping.
