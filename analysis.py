# %% [markdown]
# # IFRS 9 expected credit loss (ECL) engine
#
# An end-to-end IFRS 9 impairment calculation on the live Lending Club book at the data snapshot
# (March 2019, the latest payment date in the file). Amounts are US dollars.
#
# | Component | Approach |
# |---|---|
# | PD | A 12-month PD model (logistic, application data) ranks borrowers; an empirical default term structure by months on book, per grade, sets the level and shape. 12-month and lifetime PD are conditional on how long each loan has already survived. |
# | Staging | Stage 3: Lending Club "Default" status, or 31-120 days late with 3+ months since the last payment. Stage 2: 31+ days past due, or a significant increase in credit risk (SICR): the 12-month PD at the reporting date (re-scored with the latest FICO) is at least double, and at least 1 point above, the 12-month PD expected at origination for the same months on book. |
# | LGD | Empirical, from charge-offs whose recoveries have had at least 24 months to come in. |
# | EAD | Scheduled amortising balance, month by month over the remaining term. |
# | ECL | Sum over future months of survival x marginal PD x LGD x EAD, discounted at the loan's interest rate: 12 months for Stage 1, lifetime for Stage 2, balance x LGD for Stage 3. |
# | Scenarios | Illustrative judgemental PD overlays (not a macroeconomic model), probability-weighted. |
#
# Default is dated at last payment plus four months (about 90 days past due), because the file has no
# delinquency history. Data: Lending Club accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`);
# set `LENDING_CLUB_CSV` to the file.

# %%
import warnings; warnings.filterwarnings("ignore")
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

import lending_club as lc

OUT = "."; FIG = f"{OUT}/figures"; DAT = f"{OUT}/data"
os.makedirs(FIG, exist_ok=True); os.makedirs(DAT, exist_ok=True)
NAVY="#16314E"; TEAL="#1C8C7D"; AMBER="#C8862B"; RED="#B23A48"; SLATE="#5B6B7B"; GREEN="#2E7D5B"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#9AA7B5","axes.linewidth":0.8,
    "axes.grid":True,"grid.color":"#E2E8EF","grid.linewidth":0.7,"axes.axisbelow":True,"figure.dpi":120})
usd = lambda x,_=None: f"${x/1e9:,.2f}bn" if abs(x)>=1e9 else (f"${x/1e6:,.0f}m" if abs(x)>=1e6 else f"${x/1e3:,.0f}k")

SICR_RATIO = 2.0         # reporting 12m PD must be at least this multiple of the origination expectation
SICR_ABS = 0.01          # ... and at least this many PD points above it
STAGE3_MONTHS = 3        # months since last payment treated as 90+ days past due
LGD_WORKOUT = 24         # months of recovery history required for a charge-off to enter the LGD estimate
OVERLAY = {"Upside": 0.85, "Base": 1.00, "Downside": 1.40}   # judgemental PD multipliers
WEIGHT = {"Upside": 0.25, "Base": 0.45, "Downside": 0.30}
MAXT = 60

USE = ['loan_amnt','funded_amnt','int_rate','installment','out_prncp','total_rec_prncp','recoveries',
       'collection_recovery_fee','grade','fico_range_low','fico_range_high','last_fico_range_low',
       'last_fico_range_high','annual_inc','dti','emp_length','home_ownership','purpose','verification_status',
       'revol_util','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','total_acc','earliest_cr_line']
EMP = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,
       '7 years':7,'8 years':8,'9 years':9,'10+ years':10}

print("Reading loan tape ...")
df = lc.load(usecols=USE)
SNAP = df.attrs["snapshot_m"]
print(" ", lc.describe(df))
num = lambda c: pd.to_numeric(df[c], errors="coerce")
df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.replace('%',''), errors='coerce')
df['fico'] = (num('fico_range_low') + num('fico_range_high')) / 2
lf = (num('last_fico_range_low') + num('last_fico_range_high')) / 2
df['last_fico'] = lf.where(lf >= 300)                       # 0 means no later pull
df['annual_inc'] = num('annual_inc')
df['payment_to_income'] = (num('installment')*12 / df['annual_inc'].replace(0, np.nan)).clip(0, 2)
df['loan_to_income'] = (num('loan_amnt') / df['annual_inc'].replace(0, np.nan)).clip(0, 10)
df['log_income'] = np.log1p(df['annual_inc'].clip(0, 1e6))
df['dti'] = num('dti').clip(-1, 100)
df['revol_util'] = pd.to_numeric(df['revol_util'].astype(str).str.replace('%',''), errors='coerce').clip(0, 200)
df['emp_length_num'] = df['emp_length'].map(EMP)
df['credit_history_yrs'] = (df['issue_m'] - lc.month_index(df['earliest_cr_line'])) / 12

NUM = ['fico','dti','log_income','payment_to_income','loan_to_income','term_months','emp_length_num',
       'revol_util','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','total_acc','credit_history_yrs']
CAT = ['home_ownership','purpose','verification_status']
for c in NUM: df[c] = pd.to_numeric(df[c], errors='coerce')
for c in CAT: df[c] = df[c].fillna('missing').astype(str)

# %% [markdown]
# ## 1. 12-month PD model
# A logistic regression on application data predicts default within 12 months of issue. It is trained on loans
# issued up to 2015 and tested on 2016-2017 issues, using only loans whose 12-month window had passed by the
# snapshot. Lending Club's own grade and interest rate are left out of the features. The model's job here is
# to rank borrowers within a grade; the PD level comes from the observed term structure in section 2.

# %%
# =====================================================================
# 1) 12-MONTH PD MODEL  (fixed window, out-of-time validation)
# =====================================================================
obs12, bad12 = lc.window_outcome(df, 12)
df['obs_12'] = obs12; df['bad_12'] = bad12
pop = df[df['obs_12']].copy()
med = pop[NUM].median()
tr = pop[pop['issue_year'] <= 2015]; te = pop[pop['issue_year'] >= 2016]
print(f"12m PD: train {len(tr):,} loans issued <=2015 ({tr['bad_12'].mean():.2%} default) | "
      f"out-of-time {len(te):,} issued 2016-{int(te['issue_year'].max())} ({te['bad_12'].mean():.2%})")
pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown='ignore', min_frequency=0.01), CAT)])
pd_model = Pipeline([("p", pre), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
pd_model.fit(tr[NUM+CAT].fillna(med), tr['bad_12'])
te_p = pd_model.predict_proba(te[NUM+CAT].fillna(med))[:, 1]
auc = roc_auc_score(te['bad_12'], te_p); gini = 2*auc - 1
fpr, tpr, _ = roc_curve(te['bad_12'], te_p); ks = float(np.max(tpr - fpr))
po_ratio = float(te_p.mean() / te['bad_12'].mean())
print(f"  out-of-time AUC {auc:.3f}  Gini {gini:.3f}  KS {ks:.3f}  predicted/observed {po_ratio:.2f}")

# %% [markdown]
# ## 2. Default term structure
# For each grade, the monthly default rate by months on book: defaults in that month divided by loans still on
# book at its start. Loans leave the risk set when they default, repay in full, or reach the last month that is
# observable at the snapshot. Months with fewer than 200 loans at risk borrow the pooled shape, scaled to the grade.

# %%
# =====================================================================
# 2) DEFAULT TERM STRUCTURE  (marginal default rate by months on book, per grade)
# =====================================================================
# Each loan is followed until it defaults, repays in full, or reaches the last month whose
# outcome is observable (snapshot minus the charge-off lag); hazard = defaults / loans at risk.
horizon = (SNAP - lc.SETTLE_MONTHS - df['issue_m']).clip(lower=0)
paid = df['loan_status'].str.contains('Fully Paid', na=False)
exit_mob = np.where(df['charged_off'], df['mob_default'],
           np.where(paid, df['last_pymnt_m'] - df['issue_m'], horizon))
exit_mob = np.minimum(np.nan_to_num(exit_mob, nan=0), np.minimum(horizon, df['term_months'])).astype(int)
is_def = (df['charged_off'] & (df['mob_default'] <= horizon) & (df['mob_default'] <= df['term_months'])).values
grades = sorted(g for g in df['grade'].dropna().unique())
def hazard_curve(mask):
    em = exit_mob[mask]; dm = exit_mob[mask & is_def]
    exits = np.bincount(np.clip(em, 0, MAXT), minlength=MAXT+1)
    at_risk = exits[::-1].cumsum()[::-1]             # loans still on book at the start of each month
    defs = np.bincount(np.clip(dm, 0, MAXT), minlength=MAXT+1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(at_risk >= 200, defs / at_risk, np.nan)
    h[0] = 0.0
    return h, at_risk
H_pool, _ = hazard_curve(np.ones(len(df), bool))
H_pool = pd.Series(H_pool).ffill().fillna(0).values
H = {}
for g in grades:
    h, _ = hazard_curve((df['grade'] == g).values)
    h = pd.Series(h)
    scale = np.nanmedian((h / pd.Series(H_pool)).replace([np.inf, -np.inf], np.nan).iloc[1:25]) if h.notna().sum() > 3 else 1.0
    H[g] = h.fillna(pd.Series(H_pool) * scale).values      # thin late months: pooled shape at the grade's level

# %% [markdown]
# ## 3. PD for the live book
# Each live loan's relative risk is its model PD divided by its grade's average. That multiplies the grade's
# term structure from the loan's current age onwards, giving a 12-month and a lifetime PD conditional on
# survival so far. The same is done with the origination data, to get the 12-month PD that was expected at
# origination for a loan of this age; the reporting-date version uses the borrower's latest FICO.

# %%
# =====================================================================
# 3) ACTIVE BOOK: PD at origination and at the reporting date
# =====================================================================
act = df[df['active']].copy()
n_act = len(act)
act['age'] = (SNAP - act['issue_m']).clip(lower=0).astype(int)
act['rem'] = (act['term_months'] - act['age']).clip(lower=1, upper=MAXT).astype(int)
X_orig = act[NUM+CAT].copy(); X_orig[NUM] = X_orig[NUM].fillna(med)
X_rep = X_orig.copy(); X_rep['fico'] = act['last_fico'].fillna(act['fico']).fillna(med['fico'])
p_orig = pd_model.predict_proba(X_orig)[:, 1]
p_rep = pd_model.predict_proba(X_rep)[:, 1]
# relative risk vs the grade's average 12m PD in the training population
grade_mean = tr.assign(p=pd_model.predict_proba(tr[NUM+CAT].fillna(med))[:, 1]).groupby('grade')['p'].mean()
gm = act['grade'].map(grade_mean).fillna(float(tr['bad_12'].mean())).values
k_orig = p_orig / gm; k_rep = p_rep / gm
HTAB = np.vstack([H[g] for g in grades] + [H_pool])                               # grades x (MAXT+1)
gcode = act['grade'].map({g: i for i, g in enumerate(grades)}).fillna(len(grades)).astype(int).values
age = act['age'].values; rem = act['rem'].values
def pd_over(k, months, mult=1.0):
    """PD over the next `months` from the reporting date, conditional on survival to `age`."""
    s = np.ones(n_act)
    for t in range(1, MAXT+1):
        m = np.minimum(age + t, MAXT)
        h = np.clip(HTAB[gcode, m] * k * mult, 0, 0.99)
        live = t <= months
        s = np.where(live, s * (1 - h), s)
    return 1 - s
act['pd12_expected'] = pd_over(k_orig, np.minimum(12, rem))
act['pd12'] = pd_over(k_rep, np.minimum(12, rem))
act['pd_life'] = pd_over(k_rep, rem)

# %% [markdown]
# ## 4. Staging
# Lending Club's "Late (31-120 days)" status covers loans one to four months behind. Those with three or more
# months since their last payment are treated as 90+ days past due and go to Stage 3 with the "Default" status
# (121+ days); the rest go to Stage 2 under the 30-days-past-due backstop. Stage 2 also takes performing loans
# that meet the SICR test. Everything else is Stage 1.

# %%
# =====================================================================
# 4) STAGING
# =====================================================================
since_pay = (SNAP - act['last_pymnt_m'].fillna(act['issue_m'])).values
status = act['loan_status'].values
dpd30 = np.isin(status, ['Late (31-120 days)', 'Default'])
stage3 = (status == 'Default') | ((status == 'Late (31-120 days)') & (since_pay >= STAGE3_MONTHS))
sicr = (act['pd12'] >= SICR_RATIO * act['pd12_expected']) & (act['pd12'] - act['pd12_expected'] >= SICR_ABS)
act['stage'] = np.where(stage3, 3, np.where(dpd30 | sicr, 2, 1))
print("Staging:", act['stage'].value_counts().sort_index().to_dict(),
      f"| SICR-triggered (not 30+ dpd) {int((sicr & ~dpd30).sum()):,} | latest FICO available {act['last_fico'].notna().mean():.0%}")

# %% [markdown]
# ## 5. Loss given default
# For each charge-off, LGD = 1 - net recoveries / principal outstanding at default. Recoveries keep arriving for
# years, so only charge-offs whose default date is at least 24 months before the snapshot are averaged; more
# recent ones would overstate LGD. LGD is applied by loan term.

# %%
# =====================================================================
# 5) LGD  (charge-offs with at least 24 months of recovery history)
# =====================================================================
co = df[df['charged_off']].copy()
co['ead_at_default'] = (num('funded_amnt').loc[co.index] - num('total_rec_prncp').loc[co.index]).clip(lower=0)
co = co[co['ead_at_default'] > 0]
co['net_recovery'] = (num('recoveries').loc[co.index].fillna(0) - num('collection_recovery_fee').loc[co.index].fillna(0)).clip(lower=0)
co['lgd'] = (1 - co['net_recovery'] / co['ead_at_default']).clip(0, 1)
co['default_year'] = (co['default_m'] // 12).astype(int)
matured = co[co['default_m'] <= SNAP - LGD_WORKOUT]
lgd_mean = float(matured['lgd'].mean()); lgd_all = float(co['lgd'].mean())
lgd_by_term = matured.groupby('term_months')['lgd'].mean().to_dict()
lgd_by_year = co.groupby('default_year')['lgd'].mean()
print(f"LGD: {lgd_mean:.3f} on {len(matured):,} charge-offs with >= {LGD_WORKOUT} months of recoveries "
      f"(all charge-offs: {lgd_all:.3f}, n={len(co):,})")
act['lgd'] = act['term_months'].map(lgd_by_term).fillna(lgd_mean).values

# %% [markdown]
# ## 6. Expected credit loss
# Month by month over each loan's remaining schedule: probability of surviving to the month x default rate in the
# month x LGD x scheduled balance, discounted at the loan's monthly rate. Stage 1 sums the first 12 months,
# Stage 2 the whole remaining life, and Stage 3 takes balance x LGD.

# %%
# =====================================================================
# 6) ECL  (month-by-month over the remaining schedule)
# =====================================================================
bal0 = act['out_prncp'].astype(float).values
inst = pd.to_numeric(act['installment'], errors='coerce').fillna(0).values
r_m = (act['int_rate'].fillna(act['int_rate'].median()).clip(3, 35).values / 100) / 12
lgd = act['lgd'].values
def ecl_for(stage, mult=1.0, k=k_rep):
    s = np.ones(n_act); bal = bal0.copy(); e12 = np.zeros(n_act); elife = np.zeros(n_act)
    for t in range(1, MAXT+1):
        m = np.minimum(age + t, MAXT)
        h = np.clip(HTAB[gcode, m] * k * mult, 0, 0.99)
        live = t <= rem
        loss = np.where(live, s * h * lgd * bal / (1 + r_m) ** t, 0.0)
        elife += loss; e12 += np.where(t <= 12, loss, 0.0)
        s = np.where(live, s * (1 - h), s)
        bal = np.maximum(bal * (1 + r_m) - inst, 0.0)
    return np.where(stage == 1, e12, np.where(stage == 2, elife, bal0 * lgd))
act['ECL'] = ecl_for(act['stage'].values)
def book(a):
    g = a.groupby('stage').agg(n=('ECL','size'), EAD=('out_prncp','sum'), ECL=('ECL','sum'))
    g['coverage'] = g['ECL'] / g['EAD']; return g.reindex([1, 2, 3]).fillna(0)
B = book(act); tot_ead = float(act['out_prncp'].sum()); tot_ecl = float(act['ECL'].sum())
print("IFRS 9 staged book:"); print(B.assign(EAD=lambda d: (d.EAD/1e6).round(1), ECL=lambda d: (d.ECL/1e6).round(2),
                                          coverage=lambda d: (d.coverage*100).round(2)).to_string())
print(f"  TOTAL EAD {usd(tot_ead)}  ECL {usd(tot_ecl)}  coverage {tot_ecl/tot_ead*100:.2f}%")

# %% [markdown]
# ### Scenarios and sensitivities
# The scenarios multiply every PD by an illustrative overlay (0.85 upside, 1.00 base, 1.40 downside) and weight
# the results 25/45/30. These are judgemental, not derived from a macroeconomic model. The sensitivities show
# how much the allowance depends on two judgements: where loans 31-120 days late are staged, and how big a rise
# in PD counts as significant.

# %%
# scenarios: judgemental overlays on PD, probability-weighted
scn = {k: float(ecl_for(act['stage'].values, mult=m).sum()) for k, m in OVERLAY.items()}
weighted = sum(WEIGHT[k] * scn[k] for k in scn)
print("Scenario ECL (judgemental PD overlays):", {k: usd(v) for k, v in scn.items()}, "| weighted", usd(weighted))

# sensitivity: treatment of the 31-120 dpd bucket, and the SICR multiple
alt3 = np.where(dpd30, 3, np.where(sicr, 2, 1)); alt2 = np.where(status == 'Default', 3, np.where(dpd30 | sicr, 2, 1))
sens_bucket = {"all 31-120 dpd in Stage 3": float(ecl_for(alt3).sum()), "base (3+ months unpaid in Stage 3)": tot_ecl,
               "all 31-120 dpd in Stage 2": float(ecl_for(alt2).sum())}
ratios = [1.5, 2.0, 2.5, 3.0, 4.0]; s2_share = []; ecl_sicr = []
for q in ratios:
    sq = (act['pd12'] >= q*act['pd12_expected']) & (act['pd12'] - act['pd12_expected'] >= SICR_ABS)
    st = np.where(stage3, 3, np.where(dpd30 | sq, 2, 1))
    s2_share.append(float(act['out_prncp'][st == 2].sum() / tot_ead * 100)); ecl_sicr.append(float(ecl_for(st).sum()))

# %% [markdown]
# ## 7. Figures
# ### 12-month PD: calibration and discrimination, out of time

# %%
# =====================================================================
# FIGURES
# =====================================================================
frac, mean_p = calibration_curve(te['bad_12'], te_p, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot([0, 1], [0, 1], '--', color=SLATE, lw=1); ax[0].plot(mean_p, frac, 'o-', color=NAVY, lw=2, ms=5)
lim = max(frac.max(), mean_p.max()) * 1.1; ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
ax[0].set_xlabel("Predicted 12-month PD"); ax[0].set_ylabel("Observed 12-month default rate")
ax[0].set_title(f"12m PD calibration, out of time (P/O {po_ratio:.2f})")
ax[1].plot(fpr, tpr, color=TEAL, lw=2, label=f"AUC {auc:.3f} / KS {ks:.3f}"); ax[1].plot([0, 1], [0, 1], '--', color=SLATE, lw=1)
ax[1].fill_between(fpr, fpr, tpr, color=TEAL, alpha=0.12); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("12m PD discrimination, out of time"); ax[1].legend(loc="lower right", frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG}/01_pd_validation.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### Loss given default

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].hist(matured['lgd'], bins=40, color=AMBER, edgecolor="white", alpha=0.9)
ax[0].axvline(lgd_mean, color=NAVY, lw=2, ls='--', label=f"LGD {lgd_mean:.2f}")
ax[0].set_xlabel("Loss given default"); ax[0].set_ylabel("Charge-offs"); ax[0].legend(frameon=False)
ax[0].set_title(f"LGD, charge-offs with {LGD_WORKOUT}+ months of recoveries")
ax[1].bar(lgd_by_year.index.astype(str), lgd_by_year.values, color=[NAVY if y*12 + 11 <= SNAP-LGD_WORKOUT else "#9AA7B5" for y in lgd_by_year.index])
ax[1].set_ylim(max(0, lgd_by_year.min()-0.05), 1); ax[1].set_title("LGD by default year (grey: recoveries incomplete)")
ax[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(f"{FIG}/02_lgd_distribution.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### Exposure and loan count by stage

# %%
cols = [GREEN, AMBER, RED]; labels = ["Stage 1", "Stage 2", "Stage 3"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].bar(labels, B['EAD'].values, color=cols, edgecolor="white"); ax[0].yaxis.set_major_formatter(FuncFormatter(usd)); ax[0].set_title("Exposure at default by stage")
ax[1].bar(labels, B['n'].values, color=cols, edgecolor="white"); ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:,.0f}k")); ax[1].set_title("Number of loans by stage")
plt.tight_layout(); plt.savefig(f"{FIG}/03_staging_portfolio.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### ECL and coverage by stage

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].bar(labels, B['ECL'].values, color=cols, edgecolor="white"); ax[0].yaxis.set_major_formatter(FuncFormatter(usd)); ax[0].set_title("ECL by stage")
ax[1].bar(labels, B['coverage'].values*100, color=cols, edgecolor="white"); ax[1].set_ylabel("%"); ax[1].set_title("Coverage (ECL / EAD) by stage")
for i, v in enumerate(B['coverage'].values*100): ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/04_ecl_by_stage.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### ECL under the scenario overlays

# %%
fig, ax = plt.subplots(figsize=(8.2, 4.6)); order = list(OVERLAY)
bars = ax.bar(order, [scn[k] for k in order], color=[GREEN, NAVY, RED], edgecolor="white")
ax.axhline(weighted, color=AMBER, lw=2, ls='--', label=f"probability-weighted {usd(weighted)}")
ax.yaxis.set_major_formatter(FuncFormatter(usd)); ax.set_title("ECL under illustrative judgemental scenario overlays")
for b, k in zip(bars, order): ax.text(b.get_x()+b.get_width()/2, scn[k], f"{usd(scn[k])}\n(PD x{OVERLAY[k]:.2f}, w {WEIGHT[k]:.0%})", ha="center", va="bottom", fontsize=8.5)
ax.legend(frameon=False, loc="upper left"); plt.tight_layout(); plt.savefig(f"{FIG}/05_macro_scenarios.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### Default term structure by grade

# %%
fig, ax = plt.subplots(figsize=(9, 4.6))
for g, col in zip(['A', 'C', 'E'], [GREEN, NAVY, RED]):
    if g in H: ax.plot(range(1, MAXT+1), np.array(H[g][1:])*100, color=col, lw=2, label=f"grade {g}")
ax.plot(range(1, MAXT+1), np.array(H_pool[1:])*100, color=SLATE, lw=1.4, ls='--', label="all grades")
ax.set_xlabel("Months on book"); ax.set_ylabel("Monthly default rate (%)"); ax.legend(frameon=False)
ax.set_title("Default term structure: marginal default rate by months on book")
plt.tight_layout(); plt.savefig(f"{FIG}/06_term_structure.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ### Sensitivity to the SICR threshold

# %%
fig, ax = plt.subplots(figsize=(8.6, 4.5)); ax2 = ax.twinx()
l1 = ax.plot([f"x{q:g}" for q in ratios], s2_share, 'o-', color=AMBER, lw=2, label="Stage 2 EAD share")
l2 = ax2.plot([f"x{q:g}" for q in ratios], np.array(ecl_sicr)/1e6, 's--', color=NAVY, lw=2, label="Total ECL")
ax.set_xlabel("SICR rule: reporting 12m PD at least this multiple of the origination expectation")
ax.set_ylabel("Stage 2 EAD share (%)", color=AMBER); ax2.set_ylabel("Total ECL ($m)", color=NAVY)
ax.set_title("ECL sensitivity to the SICR threshold"); ax.legend(handles=l1+l2, loc="upper right", frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG}/07_sicr_sensitivity.png", bbox_inches="tight"); plt.close()

# %% [markdown]
# ## Save the results
# `results.json` holds every number quoted in the README; a 50,000-loan sample of the scored book goes to
# `data/scored_portfolio_sample.csv`.

# %%
# =====================================================================
SUMMARY = dict(
 snapshot=lc.month_label(SNAP), currency="USD", n_total=int(len(df)), n_active=int(n_act),
 active_definition="Current, In Grace Period, Late (16-30), Late (31-120) or Default with principal outstanding",
 pd_model=dict(outcome="default within 12 months of issue", train=f"<=2015 (n={len(tr):,})",
               out_of_time=f"2016-{int(te['issue_year'].max())} (n={len(te):,})", auc=round(auc, 3), gini=round(gini, 3),
               ks=round(ks, 3), predicted_over_observed=round(po_ratio, 3)),
 lgd=dict(mean_matured=round(lgd_mean, 3), n_matured=int(len(matured)), mean_all_chargeoffs=round(lgd_all, 3),
          workout_months=LGD_WORKOUT),
 staging=dict(rule=f"Stage 3: Default status, or Late (31-120) with {STAGE3_MONTHS}+ months since last payment; "
                   f"Stage 2: 31+ dpd, or 12m PD >= {SICR_RATIO}x the origination expectation and +{SICR_ABS:.0%}",
              latest_fico_available=round(float(act['last_fico'].notna().mean()), 3),
              sicr_only=int((sicr & ~dpd30).sum())),
 total_ead=round(tot_ead, 0), total_ecl=round(tot_ecl, 0), coverage=round(tot_ecl/tot_ead, 4),
 stage={int(i): dict(n=int(B.loc[i, 'n']), ead=round(float(B.loc[i, 'EAD']), 0), ecl=round(float(B.loc[i, 'ECL']), 0),
                     coverage=round(float(B.loc[i, 'coverage']), 4)) for i in [1, 2, 3]},
 scenarios={k: round(v, 0) for k, v in scn.items()}, overlays=OVERLAY, weights=WEIGHT,
 scenario_note="illustrative judgemental PD overlays, not a macroeconomic model",
 weighted_ecl=round(weighted, 0),
 sensitivity_31_120_bucket={k: round(v, 0) for k, v in sens_bucket.items()},
 sensitivity_sicr={f"x{q:g}": dict(stage2_ead_pct=round(a, 2), ecl=round(b, 0)) for q, a, b in zip(ratios, s2_share, ecl_sicr)})
json.dump(SUMMARY, open(f"{OUT}/results.json", "w"), indent=2)
act[['loan_status','stage','out_prncp','pd12_expected','pd12','pd_life','lgd','ECL','int_rate','fico','last_fico']]\
   .sample(min(50000, n_act), random_state=1).round(5).to_csv(f"{DAT}/scored_portfolio_sample.csv", index=False)
print("\n=== SUMMARY ==="); print(json.dumps(SUMMARY, indent=2)); print("\nDONE: figures + results written.")

# %% [markdown]
# ## Limitations
# - **PD does not use current arrears.** The PD model and term structure use application data, age and grade,
#   not whether a loan is behind today. Loans 31-120 days late that stay in Stage 2 are therefore likely
#   under-provided; the staging sensitivity shows the range.
# - **Scenarios are judgemental overlays**, not a macroeconomic model linking PD to unemployment or GDP.
# - **No prepayment in EAD.** Balances follow the contractual schedule; many of these loans repay early, so
#   lifetime ECL is somewhat overstated.
# - **LGD ignores recovery timing.** Recoveries are not discounted back to the default date, which slightly
#   understates LGD.
# - **One snapshot, one book.** There is no movement between reporting dates, and the term structure pools
#   2007-2018 vintages, so it reflects a benign US credit period on average.
