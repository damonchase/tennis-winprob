# Tennis Win Probability

This project estimates the probability that player 1 wins a men's Grand Slam
match as it progresses point by point. It compares a PyTorch GRU with two
baselines: a pre-match Elo probability held constant throughout the match, and a
tennis scoring Markov model calibrated to the Elo pre-match probability.

## Dataset

The processed point-by-point dataset contains **4,035 men's singles matches and
906,260 points** from the 2011-2024 Grand Slam files. Matches are split
chronologically by year:

| Split | Years | Matches | Points |
| --- | --- | ---: | ---: |
| Train | 2011-2021 | 3,343 | 749,362 |
| Validation | 2022 | 225 | 51,284 |
| Test | 2023-2024 | 467 | 105,614 |

The held-out validation and test sets contain only Wimbledon and US Open
matches; Australian Open and French Open matches are absent from those splits.
In particular, the experiment provides no held-out evidence for clay-court
matches, so the results should not be generalized to clay or other tours
without further evaluation.

Point features describe the state **before** the current point. Match-level Elo
ratings are frozen before each tournament begins, preventing the match or other
matches in the same event from affecting that tournament's ratings. The GRU's
running service statistic uses only earlier point outcomes within the same
match. The final match winner is the label at every point; each test prediction
is aligned to its match and point key.

Data source: Jeff Sackmann's tennis data, distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Test results

Metrics are computed point by point on the held-out comparison set. Lower Brier
score and log loss are better; higher ROC-AUC is better.

| Model | ROC-AUC (95% CI) | Brier (95% CI) | Log loss (95% CI) |
| --- | ---: | ---: | ---: |
| GRU | 0.8310 [0.8071, 0.8546] | 0.1700 [0.1556, 0.1832] | 0.5088 [0.4721, 0.5432] |
| Markov | 0.8742 [0.8466, 0.8984] | 0.1449 [0.1299, 0.1603] | 0.4385 [0.4004, 0.4776] |
| Elo-only | 0.7612 [0.7169, 0.8073] | 0.2001 [0.1803, 0.2195] | 0.5805 [0.5318, 0.6284] |

The GRU improves on the Elo-only baseline, but **does not outperform the
Markov model overall**. Paired bootstrap resampling of matches found the GRU's
log loss was worse than Markov's by 0.0703 (95% CI: +0.0373 to +0.1020), and
its Brier score was worse by 0.0251 (95% CI: +0.0118 to +0.0374). Compared with
Elo-only, GRU log loss was lower by 0.0717 (95% CI: -0.1265 to -0.0189).

### Takeaways

- Modeling tennis scoring dynamics is a strong baseline: the Markov model
  outperformed both the GRU and Elo-only model on the complete held-out point
  set.
- The GRU beat Elo-only, indicating that in-match point and score information
  adds value beyond a fixed pre-match rating.
- The Markov model's Brier score was significantly better in each of the first
  seven 10%-wide match-progress buckets. From 70% onward, the paired intervals
  included zero; there is not clear evidence that either model is better in
  those later buckets.
- These findings apply to this dataset and split, not to every surface or
  professional tennis match. The held-out set excludes clay tournaments and
  covers only Wimbledon and the US Open.

The GRU was trained by minimizing masked pointwise binary cross-entropy, with
the checkpoint selected using validation loss. A single positive temperature
was fitted on all real validation points (**T = 0.9704**) and then applied to
test predictions. Model comparisons and confidence intervals were computed
with paired bootstrap resampling at the match level to account for correlated
points within matches.

Detailed outputs are in `reports/model_comparison.csv`,
`reports/model_differences.csv`, and `reports/where_gru_helps.csv`.

## Running the project

The core pipeline scripts are in `src/`. In order, the data preparation and
model evaluation workflow is:

1. `src/data.py` - download raw point-by-point and tour match data.
2. `src/build_dataset.py` - create the point-level dataset and match metadata.
3. `src/elo.py` - calculate tournament-frozen pre-match Elo features.
4. `src/baselines.py` - generate Elo-only and Markov predictions.
5. `src/train.py` - train the GRU and save the best validation-loss checkpoint.
6. `src/calibrate.py` - fit the validation temperature.
7. `src/evaluate_bootstrap.py` - generate calibrated test predictions.
8. `src/check_gru_file.py` - check prediction coverage and key alignment.
9. `src/compare_models.py data/processed/gru_test_preds.csv` - compare test
   metrics and write detailed reports.

The FastAPI service is `src/main.py`; the Streamlit interface is `src/app.py`.

## Data credit

Data by Jeff Sackmann (Tennis Abstract). See the source files and data
dictionary for attribution and licensing details.
