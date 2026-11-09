"""Step 1: generate the synthetic ShelfIt dataset from business-plan assumptions."""
import numpy as np, pandas as pd
rng=np.random.default_rng(42)
N=400
seg_p={'Beauty & Fragrance':.28,'Modest Fashion':.24,'Accessories & Gadgets':.2,'Home & Gifting':.16,'Packaged Food':.12}
segs=rng.choice(list(seg_p),N,p=list(seg_p.values()))
aov_base={'Beauty & Fragrance':180,'Modest Fashion':260,'Accessories & Gadgets':90,'Home & Gifting':150,'Packaged Food':75}
ret_base={'Beauty & Fragrance':.04,'Modest Fashion':.12,'Accessories & Gadgets':.06,'Home & Gifting':.05,'Packaged Food':.02}
platform=rng.choice(['Instagram','TikTok','WhatsApp','Instagram + TikTok'],N,p=[.45,.2,.15,.2])
emirate=rng.choice(['Dubai','Sharjah','Ajman','Abu Dhabi'],N,p=[.55,.25,.08,.12])
tenure=rng.integers(1,25,N)
orders=np.clip(rng.lognormal(5.3,0.8,N)*(1+0.02*tenure),5,1200).round().astype(int)
plan=np.where(orders<150,'Starter',np.where(orders<600,'Growth','Pro'))
shelves=np.where(plan=='Starter',1,np.where(plan=='Growth',rng.integers(2,5,N),rng.integers(5,11,N)))
aov=np.array([aov_base[s] for s in segs])*rng.normal(1,.2,N)
cod=np.clip(rng.normal(.6,.15,N),.1,.95)
ret=np.clip(np.array([ret_base[s] for s in segs])+rng.normal(0,.02,N),0,.4)
ontime=np.clip(rng.normal(.93,.04,N),.75,1)
tickets=rng.poisson(1+orders/200+ret*20)
vas=np.clip(rng.poisson(1.5,N)+(segs=='Home & Gifting')*2,0,10)
dash=np.clip(rng.normal(12,6,N)+(platform=='WhatsApp')*-4,0,30).round().astype(int)
remit=np.clip(rng.normal(3,1,N),1,8).round(1)
storage_fee=np.where(plan=='Starter',200,np.where(plan=='Growth',180,160))*shelves
ppf=rng.uniform(5,8,N); dm=rng.uniform(3,5,N)
rev=(storage_fee+orders*(ppf+dm)+vas*rng.uniform(40,90,N)+rng.normal(0,150,N)).round(0)
csat=np.clip(5.5+ (ontime-.93)*30 -tickets*.12 -(remit-3)*.4 +rng.normal(0,.8,N),1,10).round(1)
logit=-1.2 -0.09*tenure -0.004*orders +0.25*tickets -0.55*(csat-6) +0.35*(remit-3) -12*(ontime-.93) -0.05*dash +(plan=='Starter')*0.6
churn=(rng.random(N)<1/(1+np.exp(-logit))).astype(int)
S=pd.DataFrame(dict(seller_id=[f'S{i:04d}' for i in range(1,N+1)],segment=segs,sales_platform=platform,emirate=emirate,
 plan=plan,tenure_months=tenure,shelves_rented=shelves,monthly_orders=orders,avg_order_value_aed=aov.round(0),
 cod_share=cod.round(2),return_rate=ret.round(3),on_time_dispatch_rate=ontime.round(3),support_tickets_month=tickets,
 value_added_jobs_month=vas,dashboard_logins_month=dash,cod_remittance_days=remit,satisfaction_score=csat,
 monthly_revenue_aed=rev,churned=churn))
# baskets of services per order
svc=['Standard Packing','Branded Packing','Gift Wrap','Insert Card','COD','Same-Day Delivery','Next-Day Delivery','Fragile Handling','Temp-Controlled Storage','Return Processing','Product Photography']
rows=[]; oid=1
for _,r in S.sample(frac=1,random_state=1).iterrows():
  for _ in range(rng.integers(5,25)):
    s=r.segment; b=set()
    b.add('Branded Packing' if (r.plan!='Starter' and rng.random()<.7) or rng.random()<.2 else 'Standard Packing')
    if rng.random()<r.cod_share: b.add('COD')
    b.add('Same-Day Delivery' if rng.random()<(.5 if r.emirate=='Dubai' else .2) else 'Next-Day Delivery')
    if s=='Home & Gifting':
      if rng.random()<.75: b.add('Fragile Handling')
      if rng.random()<.6: b.add('Gift Wrap')
    if 'Gift Wrap' in b and rng.random()<.8: b.add('Insert Card')
    if s in('Beauty & Fragrance','Packaged Food') and rng.random()<.7: b.add('Temp-Controlled Storage')
    if s=='Beauty & Fragrance' and 'Branded Packing' in b and rng.random()<.5: b.add('Insert Card')
    if s!='Home & Gifting' and rng.random()<.08: b.add('Gift Wrap')
    if 'COD' in b and rng.random()<(r.return_rate*2.5): b.add('Return Processing')
    if rng.random()<.04: b.add('Product Photography')
    rows.append(dict(order_id=f'O{oid:05d}',seller_id=r.seller_id,segment=s,**{c:int(c in b) for c in svc}));oid+=1
O=pd.DataFrame(rows)
O['services_list']=O[svc].apply(lambda x:', '.join([c for c in svc if x[c]]),axis=1)
dd=pd.DataFrame([
('Sellers','seller_id','Unique seller ID',''),('Sellers','segment','Product category of seller','Clustering/classification feature'),
('Sellers','sales_platform','Main social channel',''),('Sellers','emirate','Seller base location',''),
('Sellers','plan','Starter <150, Growth 150–600, Pro 600+ orders/month','Alt. classification target'),
('Sellers','tenure_months','Months with ShelfIt',''),('Sellers','shelves_rented','Shelves rented',''),
('Sellers','monthly_orders','Avg orders/month',''),('Sellers','avg_order_value_aed','Avg customer order value (AED)',''),
('Sellers','cod_share','Share of orders paid cash-on-delivery',''),('Sellers','return_rate','Share of orders returned',''),
('Sellers','on_time_dispatch_rate','Orders dispatched within SLA',''),('Sellers','support_tickets_month','Complaints/queries per month',''),
('Sellers','value_added_jobs_month','Photography/kitting/gift jobs per month',''),('Sellers','dashboard_logins_month','Dashboard engagement',''),
('Sellers','cod_remittance_days','Days to remit COD cash to seller',''),('Sellers','satisfaction_score','CSAT 1–10',''),
('Sellers','monthly_revenue_aed','Revenue ShelfIt earns from seller per month','REGRESSION target'),
('Sellers','churned','1 = seller left ShelfIt','CLASSIFICATION target'),
('Orders','order_id','Unique order ID',''),('Orders','seller_id','Links to Sellers',''),
('Orders','[service columns]','1 if service used on that order','ASSOCIATION RULE MINING baskets'),
('Orders','services_list','Same basket as comma list','')],columns=['sheet','column','description','analysis use'])
with pd.ExcelWriter('data/ShelfIt_Simulated_Data.xlsx') as w:
  dd.to_excel(w,'Data Dictionary',index=False); S.to_excel(w,'Sellers',index=False); O.to_excel(w,'Orders',index=False)
S.to_csv('data/shelfit_sellers_generated.csv',index=False); O.to_csv('data/shelfit_orders_generated.csv',index=False)
print(S.describe().T[['mean','min','max']].round(2)); print('churn',S.churned.mean(),'orders',len(O)); print(O[svc].mean().round(2))
