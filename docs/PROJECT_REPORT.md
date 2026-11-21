# Federated Learning for Educational Analytics
## Final-year B.Tech project report - structure and draft content

> **How to use this file.** Each chapter gives (a) what examiners expect there, (b) a factual draft built from this
> project's actual code and results, and (c) `[WRITE]` markers where you must write in your own words. Keep every
> number traceable to `results/reports/`. Do not add claims that the experiments did not test. Verify every
> reference against its original source before the final submission.

---

### Front matter
Title page, certificate, declaration, acknowledgements, abstract, table of contents, list of figures, list of tables,
abbreviations (FL, FedAvg, IID, MLP, BCE, ReLU, ROC AUC, DP).

---

## Abstract (150-250 words)

Draft: Educational institutions hold student data whose pooling is restricted by privacy and governance rules, which
limits the quality of predictive models any single institution can build. Federated learning (FL) trains a shared
model while data remain in place: participants train locally and exchange only model parameters. This project
implements a transparent Federated Averaging (FedAvg) simulation with three simulated institutions on the UCI Student
Performance dataset (395 students, mathematics course), predicting Pass (final grade G3 ≥ 10) versus Fail. A
leakage-safe pipeline splits the data before fitting any preprocessing, and the same 79-student test set is used for
all comparisons. Over five random seeds the federated global model reached test accuracy 0.871 ± 0.027 with IID clients
and 0.866 ± 0.006 with label-skewed Non-IID clients, compared with 0.876 ± 0.028 for centralized training, showing no
meaningful accuracy cost for keeping data local on this dataset. An early-prediction variant without mid-year grades
performed no better than the always-Pass baseline (0.671) and is reported as a negative result. The system includes an
eight-page analytics dashboard, a schema-validated prediction tool and 62 automated tests, and the report discusses why
FedAvg alone is not a privacy guarantee.

`[WRITE]` Rephrase in your own words; keep the numbers.

---

## Chapter 1 - Introduction

1.1 Background: data-driven educational analytics; why prediction of at-risk students matters.
1.2 Privacy and governance constraints on student data.
1.3 Federated learning in one paragraph (McMahan et al., 2017).
1.4 Scope: a **simulation** on a public dataset; no real institutions.
1.5 Organisation of the report.

`[WRITE]` 2-3 pages.

## Chapter 2 - Problem definition and objectives

**Problem.** Build a predictive model of student Pass/Fail across multiple institutions without transferring raw
student records to a central location, and quantify the accuracy trade-off, including under heterogeneous (Non-IID)
institutional data.

**Objectives** (copy from README, section Objectives).

**Non-goals.** Formal privacy guarantees, real deployment, more than one dataset.

## Chapter 3 - Literature review

Write 1-2 paragraphs per source, in your own words, then a gap statement. Suggested genuine sources (verify each):

| Topic | Source | What to take from it |
|---|---|---|
| FedAvg | McMahan, Moore, Ramage, Hampson, Agüera y Arcas (2017), AISTATS | algorithm, local epochs vs rounds, first Non-IID experiments |
| Non-IID effects | Zhao, Li, Lai, Suda, Civin, Chandra (2018), arXiv:1806.00582 | accuracy loss grows with label skew; weight divergence |
| Survey | Kairouz et al. (2021), *Foundations and Trends in ML* 14(1-2) | taxonomy, open problems, privacy attacks |
| Survey | Li, Sahu, Talwalkar, Smith (2020), *IEEE Signal Processing Magazine* 37(3) | challenges: communication, heterogeneity, privacy |
| Heterogeneity method | Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith (2020), MLSys - FedProx | proximal term as a fix for Non-IID |
| Secure aggregation | Bonawitz et al. (2017), ACM CCS | server sees only the sum of updates |
| Differential privacy | McMahan, Ramage, Talwar, Zhang (2018), ICLR - DP-FedAvg; Abadi et al. (2016), ACM CCS - DP-SGD | clipping + noise; privacy budget |
| Update leakage | Zhu, Liu, Han (2019), NeurIPS - Deep Leakage from Gradients; Shokri et al. (2017), IEEE S&P - membership inference | why FedAvg alone is not private |
| Dataset / baseline study | Cortez & Silva (2008), *Proc. 5th FUBUTEC*, EUROSIS | dataset description; their finding that G1/G2 dominate prediction |
| Tools | Paszke et al. (2019), NeurIPS - PyTorch; Pedregosa et al. (2011), JMLR - scikit-learn; Kingma & Ba (2015), ICLR - Adam | cite the tools you used |

**Gap statement draft.** Most FL studies use image or text benchmarks; applications to small tabular educational data
with an explicit leakage-safe protocol and a comparison against a trivial baseline are rare. `[WRITE]`

## Chapter 4 - Dataset

Use README section *Dataset* plus the EDA notebook figures:
`results/figures/eda_target_distribution.png`, `eda_categorical_pass_rates.png`, `eda_numeric_correlations.png`,
`eda_g1_g2_vs_g3.png`, `eda_correlation_heatmap.png`. Include the attribute dictionary (from `src/data/load.py`).
State: 395 rows, 33 columns, 0 missing, 265/130 Pass/Fail, G3 = 0 for 38 students, r(G1,G3) ≈ 0.80, r(G2,G3) ≈ 0.90,
rule "pass if G2 ≥ 10" alone agrees with the label for 91.9 % of students (notebook output).
Limitations: size, two Portuguese schools, 2005-06, self-reported attributes.

## Chapter 5 - Proposed methodology

5.1 **Target definition.** Pass = G3 ≥ 10. Grade bands A-F defined in `src/config.py` for a possible five-class
extension that was **not trained** in this work.

5.2 **Two feature variants.** full (with G1, G2; 58 features) and early (without; 56 features), each a separate
experiment with its own preprocessor and models.

5.3 **Leakage prevention.** G3 removed from inputs; stratified 80/20 split *before* fitting; one-hot encoder and
standard scaler fit on the 316 training rows only; test rows transformed with the fitted objects; unit tests check that
the scaler means equal training means and that a single raw row transforms to the stored processed row.

5.4 **Client simulation.** 3 clients over the training rows only. IID: stratified random thirds (67 % pass each).
Non-IID: label skew with target pass fractions 0.90 / 0.67 / 0.43 and equal sizes (realised 91 % / 68 % / 43 %).
Coverage check: each index used exactly once. Each client reserves 20 % of its rows as a private validation slice.

5.5 **Model.** MLP 58 → 32 → 16 → 1, ReLU, BCEWithLogits, Adam 0.001, batch 16, 2,433 parameters.

5.6 **FedAvg.** Formula, worked example, pure implementation, test cases.

5.7 **Round protocol.** Broadcast, local 5 epochs, return weights + n_k + 2 losses, weighted average, repeat 20 rounds.
Server never accesses client data (enforced by a test scanning the server source).

5.8 **Evaluation protocol.** Same test set for all; metrics: accuracy, precision, recall, F1 (Pass class), Fail recall,
balanced accuracy, ROC AUC, confusion matrix; always-Pass baseline; 5 seeds; mean ± std; no model selection on test.

## Chapter 6 - System architecture and design

Figures: architecture diagram (dashboard page 3 or README ASCII), pipeline flowchart
(verify → prepare → partition → experiments → figures → dashboard), module diagram of `src/`.
Data-flow table: what stays local vs what is transmitted. Technology stack and the Flower-vs-custom decision.

## Chapter 7 - Implementation

Describe each module briefly with one code excerpt each: `fedavg.py` (aggregation), `client.py` (ClientUpdate),
`server.py` (round loop), `preprocess.py` (fit-on-train), `partition.py` (Non-IID), `predictor.py` (validation),
dashboard structure. Testing strategy (62 tests, categories). Reproducibility check (seed-42 re-run bit-identical).

## Chapter 8 - Results and discussion

8.1 Tables from `results/reports/metrics_table.md` (both variants).
8.2 Figures: `metrics_full.png`, `convergence_full.png`, `test_accuracy_full.png`, `confusion_full.png`,
`client_val_loss_full.png`, `communication_cost.png`; early-variant equivalents.
8.3 Discussion points (all supported by the files):
- FL ≈ centralized: best-worst gap 1.0 pt < std 2.8 pt.
- Non-IID: accuracy 0.866 vs 0.871 (IID), precision 0.918 vs 0.922; low variance across seeds (0.006); client-level
  validation loss highest for the 43 %-pass institution → averaged model is a compromise.
- Over-fitting: centralized validation loss minimum ≈ round 5 then rising above 1.0 by round 20; FedAvg minimum ≈ round
  10-11 then mild rise; FedAvg behaves like a regulariser here (fewer updates per sample per round, averaging).
- Early variant negative result: accuracy ≤ baseline, Fail recall 0.31-0.39, ROC AUC 0.56-0.65.
- Communication: 9.7 KB per message, 1.17 MB total vs ~45 KB raw CSV.
- Statistical caution: 79 test students; 1 student ≈ 1.3 pt.
8.4 Threats to validity: single dataset, label skew only, fixed partition seed, no hyper-parameter search.

## Chapter 9 - Conclusion and future scope

Conclusion draft: FedAvg reproduced centralized accuracy on this dataset while keeping records local; Non-IID label
skew reduced accuracy only marginally but visibly disadvantaged the minority-distribution institution; prediction is
only useful once mid-year grades exist; FedAvg alone is not a privacy guarantee. Future scope: DP-FedAvg, secure
aggregation, early stopping, FedProx/SCAFFOLD, second dataset, more clients, real transport. `[WRITE]`

## References (verify each before submission)

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, 1273-1282.
2. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-IID data. *arXiv preprint arXiv:1806.00582*.
3. Kairouz, P., et al. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2), 1-210.
4. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated learning: Challenges, methods, and future directions. *IEEE Signal Processing Magazine*, 37(3), 50-60.
5. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems (MLSys)*, 2, 429-450.
6. Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., Ramage, D., Segal, A., & Seth, K. (2017). Practical secure aggregation for privacy-preserving machine learning. *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 1175-1191.
7. McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. *International Conference on Learning Representations (ICLR)*.
8. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS)*, 308-318.
9. Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
10. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *2017 IEEE Symposium on Security and Privacy (S&P)*, 3-18.
11. Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In A. Brito & J. Teixeira (Eds.), *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, 5-12. EUROSIS.
12. Cortez, P. (2014). Student Performance [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/320/student+performance
13. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
14. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
15. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

## Appendices

A. Hyper-parameters (`results/reports/experiment_config.json`). B. Per-seed results (`results/reports/runs/`).
C. Test list (`python -m pytest -v`). D. Dashboard screenshots (`docs/screenshots/`). E. Attribute dictionary.
