import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
R=ROOT+"/results/phase1_pilot"; L=np.load(ROOT+"/results/labels_phase1.npz"); eff=pd.read_csv(f"{R}/efficiency.csv").set_index("model")
MODELS=list(eff.index); SEEDS=[0,1,2]
def auc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def pred(mo,s,k):
    f=np.load(f"{R}/preds/{mo.replace(' ','_')}_s{s}.npz"); return np.r_[f["Shenzhen"],f["Montgomery"]] if k=="SZ+MC" else f[k]
Y={"TBX11K-test":L["TBX11K-test"],"Shenzhen":L["Shenzhen"],"Montgomery":L["Montgomery"],"SZ+MC":np.r_[L["Shenzhen"],L["Montgomery"]]}
A=pd.DataFrame([(mo,s,k,auc(Y[k],pred(mo,s,k))) for mo in MODELS for s in SEEDS for k in Y],columns=["model","seed","set","auc"])
# Fig 1: seed variance
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=False)
for ax,k in zip(axes,["TBX11K-test","Shenzhen","Montgomery"]):
    for i,mo in enumerate(MODELS):
        v=A[(A.model==mo)&(A.set==k)].auc.values
        ax.scatter([i]*3+np.array([-.08,0,.08]),v,s=22,zorder=3); ax.hlines(v.mean(),i-.25,i+.25,color="k",lw=1.5,zorder=2)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([m.replace(" (ref.)","*") for m in MODELS],rotation=40,ha="right",fontsize=8)
    ax.set_title(k+(" (internal)" if k.startswith("TBX") else " (external)")); ax.set_ylabel("AUC"); ax.grid(axis="y",alpha=.3)
    if not k.startswith("TBX"): ax.axhline(.5,ls=":",c="gray",lw=1)
fig.text(.5,-.02,"Dots = individual seeds, bar = mean. Dotted line = chance (0.5). *ResNet50 is a non-lightweight reference.",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("fig_seed_variance.png",dpi=300,bbox_inches="tight"); plt.close()
# Fig 2: trade-off with seed range
fig,ax=plt.subplots(figsize=(6.4,4.4)); off={"ShuffleNetV2":(-6,10),"MobileNetV3-L":(6,-12),"EfficientNet-Lite0":(6,-12)}
for mo in MODELS:
    v=A[(A.model==mo)&(A.set=="SZ+MC")].auc.values; x=eff.loc[mo,"lat_ms_1t_median"]
    ax.errorbar(x,v.mean(),yerr=[[v.mean()-v.min()],[v.max()-v.mean()]],fmt="o",capsize=3,ms=6)
    ax.annotate(f"{mo.replace(' (ref.)','*')}\n{eff.loc[mo,'GFLOPs']:.2f} GFLOPs",(x,v.mean()),xytext=off.get(mo,(6,4)),textcoords="offset points",fontsize=7)
ax.axhline(.5,ls=":",c="gray",lw=1); ax.set_ylim(.4,.85); ax.set_xlabel("CPU latency, 1 thread (ms, median)"); ax.set_ylabel("External AUC (Shenzhen + Montgomery)")
ax.set_title("External AUC vs. CPU latency (bars = seed min-max)",fontsize=10); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_tradeoff.png",dpi=300); plt.close()
# Main table
t=[]
for mo in MODELS:
    r={"Model":mo.replace(" (ref.)","*"),"Params (M)":eff.loc[mo,"params_M"],"GFLOPs":eff.loc[mo,"GFLOPs"],"Size (MB)":eff.loc[mo,"size_MB"],
       "CPU ms 1T":eff.loc[mo,"lat_ms_1t_median"],"CPU ms 4T":eff.loc[mo,"lat_ms_4t_median"]}
    for k,lab in [("TBX11K-test","AUC internal"),("Shenzhen","AUC Shenzhen"),("Montgomery","AUC Montgomery"),("SZ+MC","AUC ext. pooled")]:
        v=A[(A.model==mo)&(A.set==k)].auc.values; r[lab]=f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
    t.append(r)
T=pd.DataFrame(t).round(2); T.to_csv("table_main.csv",index=False); print(T.to_string(index=False))
