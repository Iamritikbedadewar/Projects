# Results and Reproducibility Checklist

Before accepting a run:

- [ ] The fixed mapping is used: abuse = harmful and Mongo pairs = harmless.
- [ ] Dataset counts are exactly 100 harmful and 100 harmless.
- [ ] No duplicate normalized inputs are reported.
- [ ] All harmful records have a gold rationale.
- [ ] All three models use the same test folds.
- [ ] The zero-shot prompt was fixed before test evaluation.
- [ ] Fine-tuning starts from the same public FLAN-T5 checkpoint in every fold.
- [ ] Five folds completed.
- [ ] BERTScore was not skipped in the final run.
- [ ] `MODEL_MANIFEST.json` records resolved public-model revisions.
- [ ] Harmful precision, recall, F1, and macro F1 are reported.
- [ ] Unsafe-compliance and over-refusal rates are reported.
- [ ] End-to-end and conditional BERTScore are distinguished.
- [ ] Mean and standard deviation across folds are reported.
- [ ] Error analysis prioritizes false negatives.
- [ ] No raw provider example appears in tables, plots, report, or slides.
- [ ] Limitations are discussed.
- [ ] The privacy audit passes.
- [ ] The module name and graded/ungraded choice are stated in the report.
