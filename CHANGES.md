# Changes

## Rebuilt PD, staging and LGD on corrected definitions (Nov 2026)

**Why.** The earlier engine had several problems a reviewer would raise:
1. **PD fitted on finished loans only.** Loans that default finish early, so recent vintages over-represented defaults (portfolio default rate 20%). The lifetime PD was then turned into a 12-month PD with a constant-hazard assumption, ignoring how long each loan had already been on book.
2. **Staging.** Stage 3 included every loan 31+ days past due; Stage 2 included loans in a grace period, and its PD trigger flagged the worst 15% of the live book by lifetime PD (from application data) rather than comparing each loan's current risk with its own origination risk.
3. **LGD.** Averaged over all charge-offs, including recent ones whose recoveries were still coming in.
4. **Scenarios** were described as macroeconomic but were judgemental multipliers.

**What changed.**
- `lending_club.py` (new, shared with the other credit projects) dates default at last payment plus four months, records the snapshot (March 2019) and computes observable outcome windows.
- 12-month PD model on a fixed 12-month window (train up to 2015, test 2016-2017), used to rank borrowers within a grade.
- Empirical default term structure by grade and months on book; 12-month and lifetime PD conditional on each loan's age.
- SICR compares the reporting-date 12-month PD (latest FICO) with the origination expectation at the same age: at least x2 and +1 point, plus the 31+ days past due backstop. Stage 3 is "Default" status or 3+ months since last payment.
- LGD from charge-offs with at least 24 months of recoveries, by term.
- EAD follows the amortisation schedule; ECL is summed month by month and discounted.
- Scenarios relabelled as illustrative judgemental overlays. Sensitivities added for the 31-120 day bucket and the SICR multiple.
- `analysis.py` is written in notebook cell format; `make_notebook.py` rebuilds and executes the notebook from it. The stale vintage figure (`06_vintage_default.png`) is removed; figure 6 is now the term structure.

**Numbers.**

| | Before | After |
|---|---|---|
| Live book | 907,864 loans, $9.510bn | 907,904 loans, $9.510bn (Lending Club "Default" status now included, as in the monitoring pack) |
| PD model, out of time | AUC 0.692 (lifetime outcome, finished loans) | AUC 0.685, P/O 0.83 (12-month outcome) |
| LGD | 0.909 (all charge-offs) | 0.899 (24+ months of recoveries) |
| Stage 1 / 2 / 3 loans | 746,242 / 140,181 / 21,441 | 868,686 / 29,999 / 9,219 |
| Coverage Stage 1 / 2 / 3 | 6.6% / 31.1% / 77.1% | 6.1% / 18.5% / 89.9% |
| Base ECL (coverage) | $1,246m (13.1%) | $700m (7.4%) |
| Scenarios upside / base / downside | $1,087m / $1,246m / $1,660m | $618m / $700m / $909m |
| Probability-weighted ECL | $1,330m | $742m |

Most of the fall comes from staging: the old Stage 2 rule put 140,181 loans ($1.8bn) on a lifetime allowance because they ranked in the worst 15% at origination, not because their risk had increased.
