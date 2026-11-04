import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A=pd.read_csv("per_run_auc.csv"); eff=pd.read_csv(ROOT+"/results/phase1_pilot"+"/efficiency.csv").set_index("model")
M=["MobileNetV3-L","EfficientNet-B0","EfficientNet-Lite0","MobileViT-S","ShuffleNetV2","SqueezeNet1.1"]
o=A[A.cond=="orig"]
# Fig 1: seed variance, 10 seeds
fig,axes=plt.subplots(1,3,figsize=(13,4)); rng=np.random.default_rng(0)
for ax,(k,t) in zip(axes,[("tbx","TBX11K-test (internal)"),("sz","Shenzhen (external)"),("mc","Montgomery (external)")]):
    for i,m in enumerate(M):
        v=o[o.model==m][k].values; ax.scatter(i+rng.uniform(-.15,.15,len(v)),v,s=16,alpha=.8,zorder=3); ax.hlines(v.mean(),i-.28,i+.28,color="k",lw=1.6,zorder=2)
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M,rotation=40,ha="right",fontsize=8); ax.set_title(t); ax.set_ylabel("AUC"); ax.grid(axis="y",alpha=.3)
    if k!="tbx": ax.axhline(.5,ls=":",c="gray",lw=1)
fig.text(.5,-.02,"Dots = individual seeds (n = 10 per model), bar = mean, dotted line = chance level.",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("fig1_seed_variance_10seeds.png",dpi=300,bbox_inches="tight"); plt.close()
# Fig 2: tradeoff with mean +- SD
fig,ax=plt.subplots(figsize=(6.4,4.4)); off={"ShuffleNetV2":(-8,9),"MobileNetV3-L":(6,-14),"EfficientNet-Lite0":(6,-14),"EfficientNet-B0":(6,6)}
for m in M:
    v=o[o.model==m].pooled; x=eff.loc[m,"lat_ms_1t_median"]
    ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt="o",capsize=3,ms=6)
    ax.annotate(f"{m}\n{eff.loc[m,'GFLOPs']:.2f} GFLOPs",(x,v.mean()),xytext=off.get(m,(6,4)),textcoords="offset points",fontsize=7)
ax.axhline(.5,ls=":",c="gray",lw=1); ax.set_ylim(.4,.85); ax.set_xlabel("CPU latency, 1 thread (ms, median)"); ax.set_ylabel("External AUC (Shenzhen + Montgomery)")
ax.set_title("External AUC vs. CPU latency (mean ± SD over 10 seeds)",fontsize=10); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig2_tradeoff_10seeds.png",dpi=300); plt.close()
# Fig 3: masking effect
w=A.pivot_table(index=["model","seed"],columns="cond",values=["tbx","pooled"])
fig,axes=plt.subplots(1,2,figsize=(12,4)); conds=["border","corners","border+corners","centercrop80"]; wd=.2
for ax,(k,t) in zip(axes,[("tbx","TBX11K-test (internal)"),("pooled","Shenzhen + Montgomery (external)")]):
    for j,c in enumerate(conds):
        d=(w[(k,c)]-w[(k,"orig")]).unstack(0)[M]
        ax.bar(np.arange(len(M))+(j-1.5)*wd,d.mean(),wd,yerr=d.std(ddof=1),capsize=2,label=c)
    ax.axhline(0,c="k",lw=.8); ax.set_xticks(range(len(M))); ax.set_xticklabels(M,rotation=40,ha="right",fontsize=8); ax.set_ylabel("Change in AUC vs. original image"); ax.set_title(t); ax.grid(axis="y",alpha=.3)
axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("fig3_masking_effect.png",dpi=300); plt.close()
# Tables
t1=[]
for m in M:
    r={"Model":m,"Params (M)":eff.loc[m,"params_M"],"GFLOPs":eff.loc[m,"GFLOPs"],"Size (MB)":eff.loc[m,"size_MB"],"CPU ms 1T":eff.loc[m,"lat_ms_1t_median"],"CPU ms 4T":eff.loc[m,"lat_ms_4t_median"]}
    for k,l in [("tbx","AUC internal"),("sz","AUC Shenzhen"),("mc","AUC Montgomery"),("pooled","AUC ext. pooled")]:
        v=o[o.model==m][k]; r[l]=f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    r["Border-masked internal AUC"]=f"{A[(A.model==m)&(A.cond=='border')].tbx.mean():.3f} ± {A[(A.model==m)&(A.cond=='border')].tbx.std(ddof=1):.3f}"
    t1.append(r)
pd.DataFrame(t1).round(2).to_csv("table1_main_10seeds.csv",index=False)
rows=[]
for k,l in [("tbx","TBX11K-test"),("sz","Shenzhen"),("mc","Montgomery"),("pooled","SZ+MC pooled")]:
    for c in conds:
        d=(w[("pooled" if k=="pooled" else k,c)]-w[(k,"orig")]).unstack(0)[M] if k in("tbx","pooled") else None
A2=A.pivot_table(index=["model","seed"],columns="cond",values=["tbx","sz","mc","pooled"])
for k,l in [("tbx","TBX11K-test"),("sz","Shenzhen"),("mc","Montgomery"),("pooled","SZ+MC pooled")]:
    for c in conds:
        d=(A2[(k,c)]-A2[(k,"orig")]).unstack(0)[M]; r={"Test set":l,"Condition":c}
        for m in M: r[m]=f"{d[m].mean():+.3f} ± {d[m].std(ddof=1):.3f}"
        rows.append(r)
pd.DataFrame(rows).to_csv("table2_masking.csv",index=False)
print(open("table1_main_10seeds.csv").read())
