# Refusing to Answer Harmful User Queries

An offline, reproducible DNLP project for classifying materials-database requests as
`harmful` or `harmless` and generating a concise refusal reason for harmful requests.

This repository contains **code only**. It does not contain, copy, transform, or upload
the confidential Matplus files. The files stay in their original local folder and are
read only while an experiment is running.

## Project requirements covered

- Three models:
  1. Kim-style CNN sentence classifier plus a policy-template refusal generator.
  2. Off-the-shelf `google/flan-t5-small` with zero-shot prompting.
  3. The same FLAN-T5 checkpoint fine-tuned locally on the project data.
- Classification evaluation: harmful-class F1, macro F1, precision, recall, false
  negative rate, false positive/over-refusal rate, and confusion matrix.
- Generation evaluation: BERTScore F1 on harmful examples.
- Reproducibility: deterministic seeds, stratified five-fold cross-validation,
  configuration file, model manifest, per-fold results, and tests.
- Analysis support: aggregate result tables, plots, private error-analysis output,
  and an ambiguous-query stress-test plan.
- Deliverables: a 4-8 page report template and a 10-minute presentation outline.

The complete mapping from the lecture slides to project files is in
[docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md).

## Confidentiality design

The project deliberately separates internet access from private-data access:

1. `download_public_models.py` downloads only public model files. It has no data-path
   argument and never opens the Matplus folder.
2. Experiment commands set Hugging Face and Transformers to offline mode.
3. The Python experiment process blocks outbound socket connections.
4. Model loading uses `local_files_only=True`.
5. Raw text is not printed or written by default.
6. Dataset folders, checkpoints, private predictions, and common cloud-sync paths are
   excluded by `.gitignore`.

Read [PRIVACY.md](PRIVACY.md) before running anything.

## Fixed project protocol

This project uses the following definitive mapping:

- `synthetic_nlq_abuse.json` -> harmful
- `synthetic_nlq_mongo_pairs.json` -> harmless
- `synthetic_nlq_ambiguous.json` -> unlabeled qualitative stress test

The 100 harmful and 100 harmless examples are combined and evaluated with stratified
five-fold cross-validation. This is the coherent interpretation of the topic slide:
the refusal task needs both classes, the abuse file contains misuse requests, and the
ordinary natural-language/MongoDB pairs provide harmless requests. The two
task-to-test-file lines in the provider README are treated as transposed documentation
lines. See [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Quick start on Windows

Requirements: Windows 10/11, Python 3.11, at least 10 GB free disk space. A CUDA GPU is
helpful for fine-tuning but not required for the CNN or zero-shot smoke test.

Open PowerShell in this project folder.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

While online, download the two public model checkpoints. This step never reads the
private dataset:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_public_models.py
```

Optionally disconnect Wi-Fi/Ethernet after that step. Inspect only the data structure;
the command prints field names and counts, never example values:

```powershell
.\scripts\run_offline.ps1 `
  -Command inspect `
  -DataDir "C:\path\to\matplus-data"
```

Run a fast baseline:

```powershell
.\scripts\run_offline.ps1 `
  -Command experiment `
  -DataDir "C:\path\to\matplus-data" `
  -Models kimcnn `
  -Folds 0
```

Run the complete five-fold comparison:

```powershell
.\scripts\run_offline.ps1 `
  -Command experiment `
  -DataDir "C:\path\to\matplus-data" `
  -Models kimcnn,flan_zero,flan_finetuned `
  -Folds 0,1,2,3,4
```

To support local error analysis, add `-SavePrivateErrors`. That output contains
confidential text and must never be uploaded or shared.

Run the two planned ablations only after the main protocol is fixed:

```powershell
# Policy-detail ablation for the zero-shot model
.\scripts\run_offline.ps1 -Command experiment -DataDir "C:\path\to\matplus-data" `
  -Models flan_zero -Folds 0,1,2,3,4 -PromptVariant short

# Gold-rationale ablation for the fine-tuned model
.\scripts\run_offline.ps1 -Command experiment -DataDir "C:\path\to\matplus-data" `
  -Models flan_finetuned -Folds 0,1,2,3,4 -RationaleMode generic
```

## Outputs

- `results_public/<run-id>/`: aggregate metrics and plots only.
- `private_outputs/<run-id>/`: checkpoints and optional raw error analysis.

Only the aggregate folder is intended for a report, and even that should be shared
only if the agreement permits derived aggregate results. Never share
`private_outputs`.

## Dataset adapter

The code recognizes common fields such as `question`, `natural_language_question`,
`nlq`, `input`, `response`, `answer`, `output`, `reason`, and nested variants. It
fails with record numbers and field names only; exception messages never include
record values.

If the structural inspection reports an incorrect inferred field, update
`question_fields` or `rationale_fields` in [configs/default.yaml](configs/default.yaml).
Do not paste a confidential example into the configuration.

## Experiment interpretation

The primary classification score is harmful-class F1. Also report harmful recall
because a false negative would process an unsafe request, and the harmless false
positive rate because it measures over-refusal.

Two BERTScore views are produced:

- **End-to-end BERTScore** assigns a score of zero to a harmful query predicted as
  harmless, so classification failures are penalized.
- **Conditional BERTScore** evaluates refusal wording only where the model correctly
  refused. It is diagnostic, not the headline generation result.

Do not claim that a high BERTScore proves factual or security correctness. Include
manual error categories and the limitations listed in the report template.

## Tests and privacy audit

The tests use invented public toy text only.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe .\scripts\privacy_audit.py
```


