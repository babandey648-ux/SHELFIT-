"""ShelfIt — Seller Analytics Dashboard (Streamlit)
Run:  streamlit run app.py   (after running the notebook, which creates data/shelfit_sellers_clean.csv)
"""
import pandas as pd, numpy as np, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

st.set_page_config(page_title='ShelfIt Analytics', layout='wide')
sns.set_theme(style='whitegrid')
PAL = ['#2E5AAC', '#E07A3F', '#3FA27A', '#C4455A', '#7E5BB5']

@st.cache_data
def load():
    s = pd.read_csv('data/shelfit_sellers_clean.csv')
    o = pd.read_csv('data/shelfit_orders_clean.csv')
    return s, o
df, orders = load()
SERVICES = [c for c in orders.columns if c not in ['order_id', 'seller_id', 'segment', 'services_list']]

# ---------- Sidebar filters ----------
st.sidebar.title('📦 ShelfIt')
st.sidebar.caption('You sell. We shelf it.')
seg = st.sidebar.multiselect('Segment', sorted(df.segment.unique()), sorted(df.segment.unique()))
plan = st.sidebar.multiselect('Plan', ['Starter', 'Growth', 'Pro'], ['Starter', 'Growth', 'Pro'])
emi = st.sidebar.multiselect('Emirate', sorted(df.emirate.unique()), sorted(df.emirate.unique()))
ten = st.sidebar.slider('Tenure (months)', int(df.tenure_months.min()), int(df.tenure_months.max()),
                        (int(df.tenure_months.min()), int(df.tenure_months.max())))
f = df[df.segment.isin(seg) & df.plan.isin(plan) & df.emirate.isin(emi) & df.tenure_months.between(*ten)]
fo = orders[orders.seller_id.isin(f.seller_id)]
if f.empty:
    st.warning('No sellers match the filters.'); st.stop()

st.title('ShelfIt — Seller Analytics Dashboard')
st.caption(f'{len(f)} of {len(df)} sellers selected · simulated data')

# ---------- KPIs ----------
k = st.columns(6)
k[0].metric('Sellers', len(f))
k[1].metric('Orders / month', f'{int(f.monthly_orders.sum()):,}')
k[2].metric('Revenue / month', f'AED {f.monthly_revenue_aed.sum()/1e6:.2f}M')
k[3].metric('Avg revenue / seller', f'AED {f.monthly_revenue_aed.mean():,.0f}', f'{f.monthly_revenue_aed.mean()-3200:+,.0f} vs plan')
k[4].metric('Churn rate', f'{f.churned.mean():.1%}', f'{(f.churned.mean()-df.churned.mean())*100:+.1f} pp vs all', delta_color='inverse')
k[5].metric('Avg CSAT', f'{f.satisfaction_score.mean():.2f}/10')

tab1, tab2, tab3, tab4 = st.tabs(['📊 Overview', '🔍 Churn diagnosis', '💰 Revenue drivers', '🎁 Services'])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(f.monthly_orders, bins=30, kde=True, ax=ax, color=PAL[0]); ax.set_title('Monthly orders per seller')
        for x in (150, 600): ax.axvline(x, ls='--', c='grey')
        st.pyplot(fig)
        st.caption('Right-skewed: most sellers do 100–400 orders/month. Dashed lines = plan thresholds (150, 600).')
    with c2:
        p = f.groupby('plan').agg(sellers=('seller_id', 'count'), revenue=('monthly_revenue_aed', 'sum')).reindex(['Starter', 'Growth', 'Pro']).fillna(0)
        sh = pd.DataFrame({'% sellers': p.sellers/p.sellers.sum(), '% revenue': p.revenue/p.revenue.sum()})
        fig, ax = plt.subplots(figsize=(6, 3.5)); sh.plot.bar(ax=ax, rot=0, color=[PAL[0], PAL[1]])
        ax.set_title('Share of sellers vs share of revenue'); st.pyplot(fig)
        st.caption('Pro sellers are a small share of sellers but a large share of revenue (Pareto effect).')
    seg_rev = f.groupby('segment').monthly_revenue_aed.agg(['count', 'sum', 'mean']).rename(columns={'count': 'sellers', 'sum': 'revenue', 'mean': 'avg/seller'})
    st.dataframe(seg_rev.style.format({'revenue': 'AED {:,.0f}', 'avg/seller': 'AED {:,.0f}'}), width='stretch')

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox('Churn rate by', ['plan', 'segment', 'emirate', 'sales_platform', 'tenure_band', 'csat_band'])
        r = f.groupby(cat, observed=True).churned.agg(['mean', 'count']).sort_values('mean')
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.barh(r.index.astype(str), r['mean'], color=PAL[3])
        ax.axvline(f.churned.mean(), ls='--', c='grey'); ax.set_title(f'Churn rate by {cat}')
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1)); st.pyplot(fig)
        if f[cat].nunique() > 1 and f.churned.nunique() > 1:
            chi2, pv, *_ = stats.chi2_contingency(pd.crosstab(f[cat], f.churned))
            st.caption(f'Chi-square = {chi2:.2f}, p = {pv:.4f} → ' + ('significant' if pv < .05 else 'not significant') + ' at 5%.')
    with c2:
        var = st.selectbox('Compare churners vs stayers on', ['satisfaction_score', 'on_time_dispatch_rate', 'tickets_per_100_orders',
                           'cod_remittance_days', 'tenure_months', 'return_rate', 'support_tickets_month'])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(x=f.churned.map({0: 'Stayed', 1: 'Churned'}), y=f[var], order=['Stayed', 'Churned'], hue=f.churned.map({0: 'Stayed', 1: 'Churned'}), palette={'Stayed': PAL[2], 'Churned': PAL[3]}, legend=False, ax=ax)
        ax.set_xlabel(''); st.pyplot(fig)
        a, b = f[f.churned == 0][var], f[f.churned == 1][var]
        if len(a) > 1 and len(b) > 1:
            t, pv = stats.ttest_ind(a, b, equal_var=False)
            st.caption(f'Stayed mean {a.mean():.2f} vs churned {b.mean():.2f} · Welch t-test p = {pv:.4f}')
    st.subheader('Early-warning list')
    st.caption('Rule from EDA: (Starter plan OR tenure ≤ 6 months) AND satisfaction ≤ 4, still active.')
    risk = f[((f.plan == 'Starter') | (f.tenure_months <= 6)) & (f.satisfaction_score <= 4) & (f.churned == 0)]
    st.dataframe(risk[['seller_id', 'segment', 'plan', 'tenure_months', 'satisfaction_score', 'on_time_dispatch_rate',
                       'cod_remittance_days', 'monthly_revenue_aed']].sort_values('satisfaction_score'), width='stretch', hide_index=True)
    st.metric('Revenue at risk / month', f'AED {risk.monthly_revenue_aed.sum():,.0f}')

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=f, x='monthly_orders', y='monthly_revenue_aed', hue='plan', hue_order=['Starter', 'Growth', 'Pro'], s=20, ax=ax)
        m, c0 = np.polyfit(f.monthly_orders, f.monthly_revenue_aed, 1)
        xs = np.linspace(0, f.monthly_orders.max(), 50); ax.plot(xs, m*xs+c0, 'k--', lw=1)
        ax.set_title(f'Revenue vs orders — slope AED {m:.2f}/order'); st.pyplot(fig)
    with c2:
        cols = ['monthly_orders', 'shelves_rented', 'value_added_jobs_month', 'avg_order_value_aed', 'satisfaction_score',
                'on_time_dispatch_rate', 'tenure_months', 'monthly_revenue_aed', 'churned']
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(f[cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, annot_kws={'size': 6}, ax=ax)
        ax.set_title('Correlation matrix'); st.pyplot(fig)
    st.subheader('What-if: revenue from one seller')
    a, b, c = st.columns(3)
    o_ = a.number_input('Orders / month', 10, 2000, 300); s_ = b.number_input('Shelves', 1, 12, 2); v_ = c.number_input('VAS jobs / month', 0, 20, 2)
    est = 216 + 10.46*o_ + 155.4*s_ + 63.6*v_
    st.success(f'Estimated monthly revenue ≈ **AED {est:,.0f}**  (linear model: 216 + 10.46×orders + 155.4×shelves + 63.6×VAS)')

with tab4:
    use = fo[SERVICES].mean().sort_values()
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4.5)); ax.barh(use.index, use.values, color=PAL[0])
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1)); ax.set_title('Service usage (% of orders)'); st.pyplot(fig)
    with c2:
        co = fo[SERVICES].T.dot(fo[SERVICES]) / max(len(fo), 1)
        lift = co / np.outer(fo[SERVICES].mean(), fo[SERVICES].mean())
        for s_ in SERVICES: lift.loc[s_, s_] = np.nan
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(lift, annot=True, fmt='.1f', cmap='RdBu_r', center=1, annot_kws={'size': 6}, ax=ax)
        ax.set_title('Co-occurrence lift (>1 = bought together)'); st.pyplot(fig)
    st.caption('Gift wrap, fragile handling and insert cards cluster together → a "Gift-Ready" bundle opportunity.')
