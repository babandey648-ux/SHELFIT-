"""Inject realistic data-quality issues into the simulated ShelfIt seller data
to mimic what an export from a WhatsApp bot / dashboard / manual sheet looks like."""
import pandas as pd, numpy as np
rng=np.random.default_rng(7)
S=pd.read_csv('data/shelfit_sellers_generated.csv')
start=pd.Timestamp('2026-10-31')
S.insert(1,'signup_date',[(start-pd.DateOffset(months=int(t))-pd.Timedelta(days=int(rng.integers(0,28)))).strftime(rng.choice(['%Y-%m-%d','%d/%m/%Y'])) for t in S.tenure_months])
S=S.drop(columns='tenure_months')
n=len(S); idx=lambda frac: rng.choice(n,int(n*frac),replace=False)
S.loc[idx(.05),'satisfaction_score']=np.nan
S.loc[idx(.03),'avg_order_value_aed']=np.nan
S.loc[idx(.04),'cod_remittance_days']=np.nan
S.loc[idx(.02),'emirate']=np.nan
S['emirate']=S.emirate.astype(object)
for i in idx(.10): S.loc[i,'emirate']=rng.choice(['dubai','DXB',' Dubai','sharjah','SHJ','Abu dhabi','AUH']) if pd.notna(S.loc[i,'emirate']) else np.nan
for i in idx(.08): S.loc[i,'segment']=str(S.loc[i,'segment']).lower().replace('&','and')
for i in idx(.06): S.loc[i,'plan']=str(S.loc[i,'plan']).upper()+' '
S['sales_platform']=S.sales_platform.replace({'Instagram + TikTok':'Instagram+TikTok'}).where(rng.random(n)>.5,S.sales_platform)
S.loc[rng.choice(n,4,replace=False),'monthly_orders']=-S.monthly_orders.iloc[:4].values
S.loc[rng.choice(n,3,replace=False),'monthly_orders']=99999
i=idx(.03); S.loc[i,'return_rate']=(S.loc[i,'return_rate']*100).round(1)   # entered as %
S.loc[rng.choice(n,3,replace=False),'satisfaction_score']=[0,99,-1]
S['monthly_revenue_aed']=['AED '+f'{v:,.0f}' if rng.random()<.3 else str(v) for v in S.monthly_revenue_aed]
S['churned']=[rng.choice(['Yes','yes','Y']) if c==1 and rng.random()<.4 else (rng.choice(['No','N']) if c==0 and rng.random()<.4 else c) for c in S.churned]
S=pd.concat([S,S.sample(12,random_state=3)]).sample(frac=1,random_state=5).reset_index(drop=True)
S.to_csv('data/shelfit_sellers_raw.csv',index=False)
O=pd.read_csv('data/shelfit_orders_generated.csv'); O=pd.concat([O,O.sample(40,random_state=2)]).reset_index(drop=True)
O.to_csv('data/shelfit_orders_raw.csv',index=False)
print(S.shape,O.shape); print(S.isna().sum()[lambda x:x>0])
