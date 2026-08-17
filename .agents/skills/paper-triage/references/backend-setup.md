# PDF backend setup

Run the preflight first:

```bash
python3 <workspace-root>/.agents/tools/paper-workspace/extract_paper.py \
  --workspace <workspace-root> \
  --check
```

Do not silently install a backend during triage. PDF parsers may download large
models and require platform-specific Torch settings. Stop before changing the
research environment; installation or model-cache changes require a separate,
explicitly authorized setup task. Test one representative paper before adopting
a backend for the corpus.

## Preferred: MinerU

Reference installation for a separate setup task (never run implicitly from
paper-triage or paper-read):

```bash
python -m pip install --upgrade pip
python -m pip install uv
uv pip install -U "mineru[all]"
mineru --help
```

MinerU's current CLI accepts a local input with `-p`, output directory with
`-o`, automatic/text/OCR parsing through `-m`, and formula/table switches. Its
hybrid backend is intended to use embedded text for born-digital PDFs while
retaining OCR for scanned content.

This workspace extractor discovers `.venv/bin/mineru` automatically and uses
`hybrid-engine` by default. When runtime capability is uncertain, test MLX/MPS
in an isolated process first. If acceleration is unavailable, select the CPU
pipeline explicitly with `--mineru-engine pipeline --mineru-device cpu`.
If Python reports `PermissionError` for `SC_SEM_NSEMS_MAX` or a macOS native
framework raises `NSException`, stop. Rerun the same extractor command in a
normal local terminal; do not patch MinerU inside `.venv`.
The first pipeline run downloads substantial model weights. If Hugging Face Xet
stalls while resuming a model, retry with `HF_HUB_DISABLE_XET=1`; do not delete
an otherwise reusable cache blindly.

Sources:

- https://opendatalab.github.io/MinerU/quick_start/
- https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/cli_tools.md

## Fallback: Marker

```bash
python -m pip install marker-pdf
marker_single --help
```

Marker supports Markdown, equations, tables, images, CPU/GPU/MPS, and optional
forced OCR. Check its model-weight license before non-research deployment.

Source: https://github.com/datalab-to/marker

## Fallback: Docling

```bash
python -m pip install docling
docling --help
```

Docling supports Markdown export, referenced images, OCR modes, tables, and
formula enrichment. Platform-specific OCR extras are optional.

Sources:

- https://docling-project.github.io/docling/getting_started/installation/
- https://docling-project.github.io/docling/reference/cli/

## Adoption test

For each candidate backend, test at least one representative two-column CV/AI
paper containing inline/display formulas, a method figure, a main result table,
and references. Compare:

- reading order;
- title/section hierarchy;
- inline and display formula preservation;
- table row/column fidelity;
- figure extraction and caption association;
- runtime, cache size, and deterministic rerun behavior.

Do not select a backend from README claims alone. Keep the normalized cache
contract stable so a later backend change only refreshes `extracted/`.
