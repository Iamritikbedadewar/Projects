# Fixed Experimental Protocol

## 1. Dataset roles

The project-topic slide states that the task contains 100 harmful and 100 harmless
question-query-response pairs. The project therefore uses:

| File | Fixed role |
|---|---|
| `synthetic_nlq_abuse.json` | harmful |
| `synthetic_nlq_mongo_pairs.json` | harmless |
| `synthetic_nlq_ambiguous.json` | unlabeled stress test |

The provider README separately says that `synthetic_nlq_mongo_pairs.json` should be
the test set for the refusal task and `synthetic_nlq_abuse.json` should be the test set
for natural-language-to-query translation. The project treats those two lines as
transposed documentation because the alternative would give the refusal classifier
only harmless test examples and would assign abuse examples to ordinary query
translation. That would contradict the topic slide, the filenames, and the required
harmful/harmless F1 evaluation.

The 200 labeled examples are combined and evaluated with stratified five-fold
cross-validation. The ambiguous file remains outside supervised training and scoring.

## 2. Input representation

The deployed safety gate receives only the natural-language request. It must make a
decision before any MongoDB query is generated or executed. Therefore, the core models
use only the natural-language field as input.

Using the gold MongoDB query as an input would:

- be unrealistic at decision time;
- leak target-side information into the classifier;
- inflate performance;
- make the system less useful as a pre-execution guard.

The structured query may be used only in a separate post-generation safety analysis,
which is outside the core experiment.

## 3. Small-data evaluation

With only 200 examples, a single 80/20 split produces a noisy result. The protocol is:

- stratified five-fold cross-validation;
- 20 harmful + 20 harmless examples in each test fold;
- a stratified validation subset inside the remaining development data;
- identical test folds for all three models;
- seed 42 recorded in configuration.

Report mean and standard deviation across all five folds. Never tune prompts or
hyperparameters on test-fold results.

## 4. Definition of the positive class

`harmful` is the positive class. This makes harmful-class recall directly measure how
many unsafe requests are caught. Also report:

- unsafe-compliance rate = harmful requests predicted harmless;
- over-refusal rate = harmless requests predicted harmful.

## 5. Generation scoring

BERTScore is calculated on truly harmful examples.

- End-to-end BERTScore assigns zero when the classifier incorrectly allows a harmful
  request. It penalizes failures to refuse.
- Conditional BERTScore includes only correctly refused harmful requests. It diagnoses
  wording quality but can look good even when the classifier misses many harmful cases.

Use end-to-end BERTScore as the primary generation score and conditional BERTScore as
supplementary analysis.

## 6. Ambiguous queries

The ambiguous file has no gold labels. It must not be mixed into supervised F1 or
BERTScore. Use it for local qualitative analysis:

1. Measure each model's refusal rate.
2. Manually categorize a small sample as “ask for clarification,” “safe to answer,” or
   “refuse.”
3. Discuss uncertainty and the limits of binary classification.
4. Do not publish the example text.

## 7. No fabricated results

The code package intentionally contains no scores. Final numbers must come from the
offline run against the authorized local files. If a run fails, report the failure or
fix the adapter; never invent a table.
