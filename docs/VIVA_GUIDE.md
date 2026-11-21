# Viva and presentation guide

Everything below is grounded in this repository's code and in `results/reports/`. Practise saying it in your own
words; do not memorise sentences. Where a number is quoted, know which file it comes from.

---

## 1. Two-minute explanation (plain language)

"Colleges collect a lot of information about their students, and a model trained on data from several colleges would
predict student outcomes better than one college alone. But student records are private, so colleges cannot simply
pool them. My project uses **federated learning** to solve this. Each institution trains the *same* small neural
network on its own students. Instead of sending the data, it sends only the model's numbers, its weights, to a central
server. The server averages the weights, giving bigger institutions more say, and sends the averaged model back. After
twenty such rounds we have one global model that learned from everyone, while no student record ever left its
institution.

I simulated three institutions using the public UCI Student Performance dataset of 395 students and predicted whether
a student passes the final exam. I compared federated training with ordinary centralized training on the same unseen
test students. The federated model reached about 87 % accuracy, the same as centralized within the noise of the
experiment, so keeping the data local cost nothing here. I also tested what happens when institutions have very
different pass rates, which is realistic, and found the averaged model still worked but served the weakest institution
least well. The whole thing runs in a dashboard where you can explore the data, watch the training, and enter a
student's details to get a prediction."

## 2. Five-minute technical explanation

1. **Data** (30 s): UCI student-mat, 395 × 33, no missing values. Target pass = G3 ≥ 10 → 265/130. Stratified 80/20
   split *first* (316/79, seed 42). One-hot + standard scaling fit on training rows only. G3 never an input. Two
   variants: full (with G1, G2, 58 features) and early (without, 56 features).
2. **Clients** (40 s): the 316 training rows split into 3 clients. IID = stratified thirds, 67 % pass each.
   Non-IID = label skew 91/68/43 % pass, equal sizes. Deterministic; every row used once; each client keeps 20 % as a
   private validation slice.
3. **Model** (30 s): MLP 58 → 32 → 16 → 1, ReLU, BCEWithLogits, Adam 0.001, batch 16, 2,433 parameters. Same class on
   every client, which is what makes averaging valid.
4. **FedAvg round** (60 s): server broadcasts weights → each client trains 5 local epochs → returns weights, n_k and
   two loss scalars → server computes Σ (n_k/N)·w_k → new global model → 20 rounds. Pure function, unit-tested with a
   hand example (100 samples at 0.8 and 300 at 0.4 → 0.5). Custom loop instead of Flower to keep it transparent and
   Windows-friendly.
5. **Experiments** (60 s): centralized (100 epochs, same budget), FedAvg IID, FedAvg Non-IID; 5 seeds; same test set.
   Full variant: 0.876 ± 0.028 / 0.871 ± 0.027 / 0.866 ± 0.006 accuracy; F1 ≈ 0.90; Fail recall ≈ 0.84. Gap smaller than
   std → indistinguishable. Always-Pass baseline 0.671. Early variant ≤ baseline → negative result. Centralized
   over-fits after round 5 (val loss 0.19 → 1.05); FedAvg's validation loss bottoms out at round 10-12 (≈0.23) and
   ends at 0.27-0.30. Non-IID: the 43 %-pass client has the highest
   validation loss.
6. **System** (30 s): schema-driven predictor (form generated from `schema.json`, same preprocessor), 8-page Streamlit
   dashboard, 62 tests, bit-identical re-run.
7. **Honesty** (30 s): simulation, not deployment; FedAvg ≠ privacy guarantee (gradient leakage, membership inference);
   future: DP, secure aggregation, early stopping.

## 3. Architecture walk-through (say while pointing at the diagram)

- Bottom: three institutions, each with a private data store and a local copy of the model.
- Top: aggregation server holding the global model. It stores no student data.
- Solid arrows up: weights + sample count. Dashed arrows down: the new global model.
- The dashed horizontal line is the trust boundary: only model parameters cross it.
- Pipeline order: verify → prepare → partition → experiments → figures → dashboard (`main.py`).

## 4. Module-by-module

| Module | What it does | Key point to say |
|---|---|---|
| `src/config.py` | all paths and hyper-parameters | single source of truth; change a value once |
| `src/data/load.py` | reads the CSV, validates 33 columns, builds targets | computed facts, nothing hard-coded |
| `src/data/preprocess.py` | split → fit on train → transform → save preprocessor + schema | leakage prevention lives here |
| `src/clients/partition.py` | IID / Non-IID index partitions | coverage check, deterministic |
| `src/clients/client.py` | `FederatedClient.train()` returns a `ClientUpdate` | data in private attributes; update = weights + n_k + 2 scalars |
| `src/models/mlp.py` | `StudentMLP`, get/set weights, parameter count | identical architecture on all clients |
| `src/models/train.py` | one training loop for everyone | BCEWithLogits + Adam |
| `src/strategies/fedavg.py` | the weighted average | pure, tested, 2-client worked example |
| `src/server/server.py` | the round loop and history | server never touches client `_X`/`_y` (test enforces) |
| `src/experiments/*` | pipeline stages | writes JSON per run |
| `src/evaluation/*` | metrics, figures, palette | mean ± std, baseline |
| `src/utils/predictor.py` | validation + inference | same preprocessor, schema-driven form |
| `dashboard/` | 8 pages | every number read from files |
| `tests/` | 62 tests | leakage, FedAvg, partitions, I/O, pages |

## 5. FedAvg numerical example (write on the board)

Two institutions, one weight. A: n = 100, w = 0.8. B: n = 300, w = 0.4. N = 400.
w_global = (100/400)·0.8 + (300/400)·0.4 = 0.20 + 0.30 = **0.50**.
Plain mean would be 0.6; FedAvg pulls toward B because B has three times the data. In the project this is done for
each of the 2,433 parameters with n_k = 84/84/83 (IID; each client's 80 % training slice, from the run files).

## 6. IID vs Non-IID practical example

IID: deal a shuffled deck of 316 students into three piles; every pile is ~67 % Pass. Each local model learns the same
thing, averaging loses nothing. Non-IID: sort by outcome first so pile A is 91 % Pass and pile C 43 % Pass. Model A
learns "almost everyone passes", model C learns "many fail"; averaging them is a compromise. We observed accuracy
0.866 vs 0.871 (tiny difference), but the global model's validation loss on institution C climbed in later rounds
(`client_val_loss_full.png`). In the literature with stronger skew the loss is much larger (Zhao et al., 2018).

## 7. Concept explanations (beginner level)

- **Why FL is useful:** more data without moving data; respects ownership; reduces breach surface.
- **Why raw data is not shared:** privacy law/policy; FL's whole point. In code, data live in `FederatedClient._X`.
- **Why weights are shared:** weights are what the model learned; they are tiny (9.7 KB) and architecture-matched so
  they can be averaged.
- **Why the global model is sent back:** so every client starts the next round from the shared knowledge and so all
  clients end with the same model.
- **How the neural network works:** 58 inputs × weights → 32 ReLU units → 16 ReLU units → 1 logit → sigmoid →
  probability of Pass. Training nudges weights to reduce cross-entropy on labelled examples.
- **Why BCEWithLogits:** binary target; combines sigmoid and cross-entropy in one numerically stable step.
- **Why Adam:** adaptive learning rates per parameter; robust default for small networks.
- **Why preprocessing:** neural networks need numbers; one-hot turns categories into 0/1 columns; scaling puts
  features on comparable ranges so training is stable. Fitting on train only avoids leaking test statistics.
- **Metrics:** accuracy = correct / all; precision (Pass) = of predicted passes, how many passed; recall (Pass) = of
  real passes, how many we caught; F1 = harmonic mean; Fail recall = of real fails, how many we caught (what a college
  cares about); confusion matrix = the 2×2 counts; ROC AUC = ranking quality independent of threshold.
- **Training vs validation vs test:** training loss is on the data the model learns from; validation loss is on each
  client's held-out slice (used only to monitor); test metrics are on the 79 untouched students and are the only
  numbers we call "performance".

## 8. Likely viva questions and answers (45)

**Project and motivation**
1. *What problem does your project solve?* Building a shared student-outcome model across institutions without
   centralising student records, and measuring the accuracy trade-off.
2. *Is this deployed in real colleges?* No. It is a simulation: three partitions of one public dataset stand in for
   institutions. I say so on the dashboard and in the report.
3. *Why federated learning rather than just anonymising the data?* Anonymisation is fragile (re-identification) and
   still requires moving records; FL moves no records at all.
4. *What is novel here?* Not the algorithm; the contribution is an end-to-end, leakage-safe, reproducible application
   of FedAvg to small tabular educational data with an honest baseline and heterogeneity study.

**Data**
5. *Describe the dataset.* UCI Student Performance, maths course, 395 students, 33 attributes, two Portuguese schools,
   2005-06; grades 0-20.
6. *Any missing values?* None; verified by code (`dataset_summary.json`).
7. *How did you define Pass/Fail?* G3 ≥ 10, the Portuguese pass mark; 265 Pass, 130 Fail.
8. *Is the dataset balanced?* No, 67 % Pass. So I report F1, Fail recall and compare to the always-Pass baseline.
9. *What is target leakage and how did you prevent it?* Using information that encodes the answer. G3 is never an
   input; and preprocessing is fit only on training rows, tested in `tests/test_preprocessing.py`.
10. *Why not use G1 and G2?* I did, in the full variant, and I also ran an early variant without them. G1/G2 make the
    model a final-outcome predictor, not an early-warning one. Without them accuracy fell to the baseline.
11. *How did you split the data?* Stratified 80/20 with seed 42 before any fitting: 316 train, 79 test.
12. *How did you encode categorical features?* One-hot encoding, `handle_unknown="ignore"`; 17 categorical columns
    become 43 binary columns (58 − 15 numeric).

**Clients and heterogeneity**
13. *How many clients and why three?* Three: enough to show aggregation and heterogeneity, small enough to keep each
    client at ~105 students.
14. *What does IID mean?* Independent and identically distributed: every client's data come from the same
    distribution, here ~67 % pass each.
15. *What does Non-IID mean and how did you create it?* Statistical heterogeneity. I used label skew: target pass
    fractions 0.90/0.67/0.43 with equal client sizes, realised 91/68/43 %.
16. *How do you know no student is in two clients?* A coverage check in `partition.py` and a unit test: the
    concatenated indices equal 0..315 exactly once.
17. *Does the test set go to any client?* No. Only the 316 training rows are partitioned.
18. *Why is each client's validation slice needed?* To monitor the global model on client-side data without ever
    touching the test set.

**Model and training**
19. *Describe your model.* MLP: 58 inputs → 32 ReLU → 16 ReLU → 1 logit; 2,433 parameters.
20. *Why such a small network?* 316 training rows; a larger network would over-fit more and would not change the
    federated story.
21. *Why must every client use the same architecture?* FedAvg averages parameters position by position; different
    shapes cannot be averaged. The code raises an error if shapes differ.
22. *What is a local epoch?* One full pass over a client's own training rows. Five per round.
23. *What is a communication round?* One cycle of broadcast → local training → upload → aggregation. Twenty in total.
24. *How did you make centralized and federated comparable?* Same model, optimiser, batch size, test set, and the
    same total passes over the data (100 epochs vs 20 × 5).
25. *Why Adam and learning rate 0.001?* Standard robust defaults for small MLPs; I did not tune hyper-parameters,
    which I state as a limitation.

**FedAvg**
26. *Write the FedAvg formula.* w = Σ_k (n_k/N)·w_k with N = Σ n_k.
27. *Why weight by n_k?* So each training sample counts equally regardless of which client holds it.
28. *Give a numerical example.* (Section 5 above.)
29. *What does the server receive from a client?* Weights (2,433 floats), n_k, training loss, validation loss. Nothing
    else; a test checks the `ClientUpdate` fields.
30. *Why not use Flower?* Flower's simulation depends on Ray, heavy on Windows, and hides aggregation. A transparent
    custom loop is better for understanding and testing; the algorithm is the same.
31. *How much data is communicated?* 9.7 KB per model message; 6 messages per round; 1.17 MB over 20 rounds.

**Results**
32. *What accuracy did you get?* Full variant, mean over 5 seeds: centralized 0.876, FedAvg IID 0.871, FedAvg Non-IID
    0.866; std 0.006-0.028; F1 ≈ 0.90; Fail recall ≈ 0.84.
33. *So FL was worse?* By 0.5-1.0 points, which is less than the seed-to-seed standard deviation and less than one test
    student (1.3 points). Statistically indistinguishable.
34. *Did Non-IID hurt?* Marginally in accuracy and precision; clearly in fairness: the 43 %-pass institution had the
    highest validation loss. With stronger skew the literature reports larger losses.
35. *Why did the centralized validation loss rise?* Over-fitting: 100 epochs on 253 rows with no early stopping.
    FedAvg's averaging acted as a regulariser; its validation loss rose only slightly.
36. *What about the early model?* Accuracy 0.656-0.676 versus baseline 0.671, Fail recall 0.31-0.39. Not usable; a
    negative result I report rather than hide.
37. *Are training losses your results?* No. Training/validation losses are shown for convergence only; all reported
    performance is on the test set.
38. *How reliable are results on 79 test students?* Limited. That is why I report mean ± std over 5 seeds and treat
    differences under ~3 points as noise.

**Privacy**
39. *Does FedAvg guarantee privacy?* No. It minimises data movement, but updates can leak (gradient inversion,
    membership inference) and the server sees each client's weights.
40. *How would you add real privacy?* Differential privacy (clip updates, add Gaussian noise; DP-FedAvg) and secure
    aggregation (server sees only the sum). Neither is implemented here.
41. *Can someone recover a student from the weights?* Attacks exist, especially with small batches and few clients;
    I make no claim that it is impossible.

**Engineering**
42. *How do you ensure the prediction form matches the model?* The form is generated from `schema.json`, saved by the
    same code that fit the preprocessor; a test checks the widget set equals the schema columns and that a raw row
    transforms to the stored processed row.
43. *How did you test the system?* 62 pytest tests: leakage, partitions, FedAvg maths, model I/O, client update
    contents, server source scan, metrics, predictor validation, every dashboard page.
44. *Is it reproducible?* Yes: fixed seeds; a seed-42 re-run reproduced all JSON results and model weights exactly.
45. *What would you do with more time?* DP-FedAvg, secure aggregation, early stopping, FedProx under stronger skew,
    the 649-student Portuguese-course file, more clients, Flower transport, and the five-class grade-band model.

## 9. Demo script (5 minutes)

1. Overview page: metrics row, architecture picture, "what it is / is not".
2. Client Distribution: toggle IID vs Non-IID.
3. Training: point at validation-loss divergence of centralized; show client-level curves for Non-IID.
4. Evaluation: table and bar chart; switch to early variant to show the red warning.
5. Prediction: default student → Predict; change G2 from 11 to 6 → re-predict; show what-if chart.
6. Privacy: the tensor table "exactly what one upload contains".
