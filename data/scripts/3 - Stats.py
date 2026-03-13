#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

INPUT = 'dataset_final_urgencias_aire.csv'
CAT_ORDER = ['buena', 'regular', 'alerta', 'preemergencia', 'emergencia']
CRIT_CATS = {'preemergencia', 'emergencia'}
WINTER_MONTHS = {4, 5, 6, 7, 8, 9}


def irr_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        'term': model.params.index,
        'IRR': np.exp(model.params.values),
        'CI95_L': np.exp(ci[0].values),
        'CI95_U': np.exp(ci[1].values),
        'p': model.pvalues.values,
    })


def bootstrap_mean_diff(x, y, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = np.mean(x) - np.mean(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boots[i] = np.mean(xb) - np.mean(yb)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return obs, lo, hi


def permutation_pvalue(x, y, n_perm=5000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = abs(np.mean(perm[:n_x]) - np.mean(perm[n_x:]))
        if stat >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def build_clusters(df):
    d = df[['FECHA', 'CAT', 'RESP_TOTAL']].copy().sort_values('FECHA').reset_index(drop=True)
    d['is_critical'] = d['CAT'].isin(CRIT_CATS)
    clusters = []
    in_cluster = False
    start_idx = None

    for i, is_crit in enumerate(d['is_critical']):
        if is_crit and not in_cluster:
            in_cluster = True
            start_idx = i
        elif not is_crit and in_cluster:
            end_idx = i - 1
            clusters.append((start_idx, end_idx))
            in_cluster = False
    if in_cluster:
        clusters.append((start_idx, len(d) - 1))

    cluster_rows = []
    event_rows = []
    cluster_id = 0

    severity_rank = {'preemergencia': 1, 'emergencia': 2}

    for start_idx, end_idx in clusters:
        length = end_idx - start_idx + 1
        if length < 2:
            continue
        pre_idx = [j for j in range(start_idx - 3, start_idx) if j >= 0]
        if len(pre_idx) < 3:
            continue
        baseline = d.loc[pre_idx, 'RESP_TOTAL'].mean()
        cats = d.loc[start_idx:end_idx, 'CAT'].tolist()
        max_cat = sorted(set(cats), key=lambda x: severity_rank.get(x, 0))[-1]
        cluster_id += 1
        start_date = d.loc[start_idx, 'FECHA']
        end_date = d.loc[end_idx, 'FECHA']
        deltas = []
        valid_lags = []
        for lag in range(0, 7):
            idx = end_idx + lag
            if idx >= len(d):
                continue
            resp = d.loc[idx, 'RESP_TOTAL']
            delta = resp - baseline
            event_rows.append({
                'cluster_id': cluster_id,
                'start_date': start_date,
                'end_date': end_date,
                'cluster_length': length,
                'cluster_max_cat': max_cat,
                'lag': lag,
                'resp_total': resp,
                'baseline_mean': baseline,
                'delta_vs_baseline': delta,
            })
            if lag >= 1:
                deltas.append(delta)
                valid_lags.append(lag)

        if len(deltas) == 0:
            continue
        peak_idx = int(np.argmax(deltas))
        peak_lag = valid_lags[peak_idx]
        peak_delta = deltas[peak_idx]
        cum_delta = float(np.sum(deltas))
        length_bin = '5+' if length >= 5 else str(length)
        cluster_rows.append({
            'cluster_id': cluster_id,
            'start_date': start_date,
            'end_date': end_date,
            'cluster_length': length,
            'cluster_max_cat': max_cat,
            'cum_delta_lag1_to_lag6': cum_delta,
            'peak_lag_1to6': peak_lag,
            'peak_delta_1to6': peak_delta,
            'length_bin': length_bin,
        })

    event_df = pd.DataFrame(event_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    return event_df, cluster_df


def main():
    df = pd.read_csv(INPUT)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df = df.sort_values('FECHA').reset_index(drop=True)
    df['CAT'] = pd.Categorical(df['CAT'], CAT_ORDER, ordered=True)
    df['CAT_ORD'] = df['CAT'].cat.codes
    if 'month' not in df.columns:
        df['month'] = df['FECHA'].dt.month
    if 'dow' not in df.columns:
        df['dow'] = df['FECHA'].dt.dayofweek
    if 't' not in df.columns:
        df['t'] = np.arange(len(df))
    if 'resp_share' not in df.columns:
        df['resp_share'] = df['RESP_TOTAL'] / df['TOTAL_URG']
    if 'is_winter' not in df.columns:
        df['is_winter'] = df['month'].isin(WINTER_MONTHS)

    # S4 descriptives by CAT
    s4 = df.groupby('CAT', observed=False).agg(
        TOTAL_URG_count=('TOTAL_URG', 'count'),
        TOTAL_URG_mean=('TOTAL_URG', 'mean'),
        TOTAL_URG_median=('TOTAL_URG', 'median'),
        RESP_TOTAL_count=('RESP_TOTAL', 'count'),
        RESP_TOTAL_mean=('RESP_TOTAL', 'mean'),
        RESP_TOTAL_median=('RESP_TOTAL', 'median'),
        resp_share_count=('resp_share', 'count'),
        resp_share_mean=('resp_share', 'mean'),
        resp_share_median=('resp_share', 'median'),
    ).reset_index()
    s4.to_csv('supp_table_s4_source.csv', index=False)

    # S5 non-parametric
    np_rows = []
    for y in ['TOTAL_URG', 'RESP_TOTAL', 'resp_share']:
        groups = [df.loc[df['CAT'] == c, y].dropna().values for c in CAT_ORDER if (df['CAT'] == c).sum() > 0]
        kw = stats.kruskal(*groups)
        rho, p = stats.spearmanr(df['CAT_ORD'], df[y], nan_policy='omit')
        np_rows.append({
            'Outcome': y,
            'Kruskal_Wallis_H': kw.statistic,
            'Kruskal_Wallis_p': kw.pvalue,
            'Spearman_rho': rho,
            'Spearman_p': p,
        })
    pd.DataFrame(np_rows).to_csv('supp_nonparametric_tests.csv', index=False)

    # S5 NB counts
    formula = 'C(CAT) + C(dow) + C(month) + t'
    count_tables = []
    for y in ['TOTAL_URG', 'RESP_TOTAL']:
        model = smf.glm(f'{y} ~ {formula}', data=df, family=sm.families.NegativeBinomial()).fit()
        tab = irr_table(model)
        tab['outcome'] = y
        count_tables.append(tab)
    pd.concat(count_tables, ignore_index=True).to_csv('supp_table_s5_counts_source.csv', index=False)

    # S5 NB rate
    dfr = df[df['TOTAL_URG'] > 0].copy()
    dfr['log_total'] = np.log(dfr['TOTAL_URG'])
    m_rate = smf.glm(
        f'RESP_TOTAL ~ {formula}',
        data=dfr,
        family=sm.families.NegativeBinomial(),
        offset=dfr['log_total']
    ).fit()
    rate_tab = irr_table(m_rate)
    rate_tab['outcome'] = 'RESP_TOTAL_rate_offset_TOTAL_URG'
    rate_tab.to_csv('supp_table_s5_rate_source.csv', index=False)

    # S1-S2 event study
    event_df, cluster_df = build_clusters(df)
    event_df.to_csv('supp_event_level_source.csv', index=False)
    cluster_df[['cluster_id','start_date','end_date','cluster_length','cluster_max_cat']].to_csv('supp_clusters_source.csv', index=False)
    cluster_df.to_csv('supp_cluster_level_source.csv', index=False)
    cluster_df.to_csv('F_cumulative_impact_by_cluster.csv', index=False)

    s1 = cluster_df.groupby(['cluster_max_cat', 'length_bin'], observed=False).agg(
        n_clusters=('cluster_id', 'count'),
        mean_cum_delta=('cum_delta_lag1_to_lag6', 'mean'),
        median_cum_delta=('cum_delta_lag1_to_lag6', 'median'),
        q25_cum_delta=('cum_delta_lag1_to_lag6', lambda x: np.quantile(x, 0.25)),
        q75_cum_delta=('cum_delta_lag1_to_lag6', lambda x: np.quantile(x, 0.75)),
        mean_peak_delta=('peak_delta_1to6', 'mean'),
        mean_peak_lag=('peak_lag_1to6', 'mean'),
    ).reset_index()
    s1.to_csv('G_cumulative_impact_by_length_severity.csv', index=False)

    s1_pub = s1.copy()
    s1_pub['Maximum severity'] = s1_pub['cluster_max_cat'].map({'emergencia': 'Emergency', 'preemergencia': 'Pre-emergency'})
    s1_pub['Cluster duration'] = s1_pub['length_bin'].map({'2': '2 days', '3': '3 days', '4': '4 days', '5+': '5+ days'})
    s1_pub['Number of clusters'] = s1_pub['n_clusters']
    s1_pub['Mean cumulative excess respiratory visits (lag 1-6)'] = s1_pub['mean_cum_delta']
    s1_pub['Median cumulative excess respiratory visits (lag 1-6)'] = s1_pub['median_cum_delta']
    s1_pub['IQR cumulative excess respiratory visits (lag 1-6)'] = s1_pub['q25_cum_delta'].round(2).astype(str) + ' to ' + s1_pub['q75_cum_delta'].round(2).astype(str)
    s1_pub['Mean peak excess respiratory visits'] = s1_pub['mean_peak_delta']
    s1_pub['Mean lag of peak excess'] = s1_pub['mean_peak_lag']
    s1_pub = s1_pub[[
        'Maximum severity', 'Cluster duration', 'Number of clusters',
        'Mean cumulative excess respiratory visits (lag 1-6)',
        'Median cumulative excess respiratory visits (lag 1-6)',
        'IQR cumulative excess respiratory visits (lag 1-6)',
        'Mean peak excess respiratory visits', 'Mean lag of peak excess'
    ]]
    s1_pub.to_csv('supp_table_s1_source.csv', index=False)

    s2 = event_df.groupby('lag', observed=False).agg(
        N_clusters=('cluster_id', 'nunique'),
        Mean_delta_vs_baseline=('delta_vs_baseline', 'mean'),
        Median_delta_vs_baseline=('delta_vs_baseline', 'median')
    ).reset_index().rename(columns={'lag': 'Lag'})
    ci_rows = []
    rng = np.random.default_rng(42)
    for lag_val in s2['Lag']:
        vals = event_df.loc[event_df['lag'] == lag_val, 'delta_vs_baseline'].dropna().values
        boots = []
        for _ in range(5000):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(sample.mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        ci_rows.append((lag_val, lo, hi))
    ci_df = pd.DataFrame(ci_rows, columns=['Lag','Bootstrap_95CI_low','Bootstrap_95CI_high'])
    s2 = s2.merge(ci_df, on='Lag', how='left')
    s2 = s2[['Lag','N_clusters','Mean_delta_vs_baseline','Median_delta_vs_baseline','Bootstrap_95CI_low','Bootstrap_95CI_high']]
    s2.to_csv('supp_table_s2_source.csv', index=False)

    # S3 seasonality
    all_days = pd.DataFrame([{
        'Group': 'All days',
        'RESP_TOTAL_n_days': df['RESP_TOTAL'].notna().sum(),
        'RESP_TOTAL_mean': df['RESP_TOTAL'].mean(),
        'RESP_TOTAL_median': df['RESP_TOTAL'].median(),
        'TOTAL_URG_n_days': df['TOTAL_URG'].notna().sum(),
        'TOTAL_URG_mean': df['TOTAL_URG'].mean(),
        'TOTAL_URG_median': df['TOTAL_URG'].median(),
        'RESP_SHARE_n_days': df['resp_share'].notna().sum(),
        'RESP_SHARE_mean': df['resp_share'].mean(),
        'RESP_SHARE_median': df['resp_share'].median(),
    }])
    winter = df[df['is_winter'] == True]
    nonwinter = df[df['is_winter'] == False]
    seasonal_rows = []
    for name, dsub in [('Winter (Apr-Sep)', winter), ('Non-winter', nonwinter)]:
        seasonal_rows.append({
            'Group': name,
            'RESP_TOTAL_n_days': dsub['RESP_TOTAL'].notna().sum(),
            'RESP_TOTAL_mean': dsub['RESP_TOTAL'].mean(),
            'RESP_TOTAL_median': dsub['RESP_TOTAL'].median(),
            'TOTAL_URG_n_days': dsub['TOTAL_URG'].notna().sum(),
            'TOTAL_URG_mean': dsub['TOTAL_URG'].mean(),
            'TOTAL_URG_median': dsub['TOTAL_URG'].median(),
            'RESP_SHARE_n_days': dsub['resp_share'].notna().sum(),
            'RESP_SHARE_mean': dsub['resp_share'].mean(),
            'RESP_SHARE_median': dsub['resp_share'].median(),
        })
    pd.concat([all_days, pd.DataFrame(seasonal_rows)], ignore_index=True).to_csv('supp_table_s3a_source.csv', index=False)

    md_resp, lo_resp, hi_resp = bootstrap_mean_diff(winter['RESP_TOTAL'], nonwinter['RESP_TOTAL'])
    md_share, lo_share, hi_share = bootstrap_mean_diff(winter['resp_share'], nonwinter['resp_share'])
    p_resp = permutation_pvalue(winter['RESP_TOTAL'], nonwinter['RESP_TOTAL'])
    p_share = permutation_pvalue(winter['resp_share'], nonwinter['resp_share'])
    s3b = pd.DataFrame([{
        'Comparison': 'Winter (Apr-Sep) vs non-winter',
        'Mean difference RESP_TOTAL': md_resp,
        'RESP_TOTAL bootstrap 95% CI low': lo_resp,
        'RESP_TOTAL bootstrap 95% CI high': hi_resp,
        'RESP_TOTAL permutation p-value': p_resp,
        'Mean difference respiratory share': md_share,
        'Respiratory share bootstrap 95% CI low': lo_share,
        'Respiratory share bootstrap 95% CI high': hi_share,
        'Respiratory share permutation p-value': p_share,
    }])
    s3b.to_csv('supp_table_s3b_source.csv', index=False)

    exclude_dates = set(event_df['end_date'])
    for _, r in cluster_df.iterrows():
        end = pd.to_datetime(r['end_date'])
        for lag in range(0, 7):
            exclude_dates.add(end + pd.Timedelta(days=lag))
        start = pd.to_datetime(r['start_date'])
        cur = start
        while cur <= end:
            exclude_dates.add(cur)
            cur += pd.Timedelta(days=1)
    df_excl = df[~df['FECHA'].isin(exclude_dates)].copy()
    winter_excl = df_excl[df_excl['is_winter'] == True]
    nonwinter_excl = df_excl[df_excl['is_winter'] == False]
    md_excl, lo_excl, hi_excl = bootstrap_mean_diff(winter_excl['RESP_TOTAL'], nonwinter_excl['RESP_TOTAL'])
    p_excl = permutation_pvalue(winter_excl['RESP_TOTAL'], nonwinter_excl['RESP_TOTAL'])
    s3c = pd.DataFrame([{
        'Comparison': 'Winter (Apr-Sep) vs non-winter excluding cluster and post-cluster days',
        'Mean difference RESP_TOTAL': md_excl,
        'Bootstrap 95% CI low': lo_excl,
        'Bootstrap 95% CI high': hi_excl,
        'Permutation p-value': p_excl,
        'N days used': len(df_excl),
    }])
    s3c.to_csv('supp_table_s3c_source.csv', index=False)

    df.to_csv('dataset_final_urgencias_aire_checked_out.csv', index=False)

if __name__ == '__main__':
    main()
