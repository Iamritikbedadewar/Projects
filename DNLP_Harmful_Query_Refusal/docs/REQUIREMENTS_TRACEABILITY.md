# Course-Requirement Traceability

| Requirement from course slides | Where it is addressed |
|---|---|
| Select one NLP task and dataset | Harmful-query refusal using the authorized Matplus data |
| Three models | KimCNN, zero-shot FLAN-T5, fine-tuned FLAN-T5 |
| Reimplemented related-work model | `src/dnlp_refusal/models/kim_cnn.py`; Kim (2014) |
| Off-the-shelf Hugging Face transformer | Local `google/flan-t5-small`, zero-shot |
| Meaningful modification to same transformer | Fold-specific local fine-tuning with joint label/rationale targets |
| Classification | `harmful` vs `harmless` |
| Text generation | Concise reason for refusing harmful input |
| F1 evaluation | Harmful precision/recall/F1 and macro F1 |
| BERTScore evaluation | RoBERTa-large layer-17 end-to-end and conditional BERTScore |
| Data finding/cleaning/preprocessing | Structure-safe adapter, validation, normalization, duplicate checks |
| Data splitting | Stratified five-fold cross-validation plus validation subset |
| Hyperparameters | `configs/default.yaml` |
| More than one evaluation aspect | F1, BERTScore, safety error rates, parse errors, manual categories |
| Ablation study | Short-policy and generic-rationale CLI variants |
| Error analysis | Optional local-only error file and category plan |
| Results interpretation | Report prompts for safety/utility trade-offs and limitations |
| Reproducibility | Seed, config, public model commit manifest, fold metrics, tests |
| 4-8 page report | `report/report_template.md` |
| 10-minute talk + discussion | `presentation/outline.md` and prepared questions |
| Module name and graded/ungraded choice | Required placeholders at top of report |
| Data confidentiality | Offline two-phase workflow, network blocking, no raw logging, privacy audit |

