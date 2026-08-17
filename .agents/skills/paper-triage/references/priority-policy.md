# Priority policy

Assign priority by qualitative judgment. Do not create a weighted score that
implies false precision.

## Core

Use `Core` when deep understanding is necessary for the active VTG/Video-LLM
program. Strong signals include:

- directly solves VTG/temporal retrieval with a Video-LLM or immediately reusable
  large-model training/post-training design;
- defines a central technical route, benchmark, protocol, or baseline needed to
  understand current results;
- is an advisor-designated anchor: UniTime, Time-R1, or TimeLens2;
- is the nearest conceptual or implementation precedent for likely project work.

Core is about dependency and research value, not prestige.

## Important

Use `Important` when the paper should be read beyond an abstract but is not a
central dependency. Examples:

- directly relevant VTG work from a non-core method family;
- a strong baseline, dataset, metric, temporal module, or training technique;
- adjacent Video-LLM temporal reasoning that transfers plausibly to VTG;
- a predecessor needed to interpret one Core paper;
- a useful contrasting route or failure mode.

## Scan

Use `Scan` when the record is useful for landscape coverage but a deep read has
low expected return now. Examples:

- only loosely related to temporal localization;
- largely duplicates a better-covered method or evidence source;
- applies a familiar technique without a reusable VTG-specific idea;
- is primarily background, a narrow application, or weakly documented.

Do not use `Scan` as a negative quality judgment.

## Decision discipline

Evaluate:

1. direct task relevance;
2. large-model relevance;
3. importance of the technical route;
4. baseline or prerequisite value;
5. dataset, metric, or SOTA-context value;
6. redundancy with already-covered papers;
7. reuse potential for future methods or implementation.

Write `Priority Reason` as one paper-specific sentence naming the decisive
feature and its value. Avoid generic phrases such as “highly relevant,” “novel,”
or “worth reading” without saying why.

Examples:

- `Core`: “Directly trains a Video-LLM to emit temporal boundaries with GRPO,
  making it a central post-training reference for the planned VTG route.”
- `Important`: “Introduces a boundary-aware matching loss used by later VTG
  systems, but it predates and does not address the current Video-LLM route.”
- `Scan`: “Evaluates generic long-video QA without temporal interval output, so
  it offers context but little reusable VTG methodology.”
