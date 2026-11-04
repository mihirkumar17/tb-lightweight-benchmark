"""Fig. 2 of the paper: external AUC vs CPU latency, legend version (needs per_run_auc.csv)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
A=pd.read_csv("per_run_auc.csv"); eff=pd.read_csv(ROOT+"/results/phase1_pilot/efficiency.csv").set_index("model")
M=["MobileNetV3-L","EfficientNet-B0","EfficientNet-Lite0","MobileViT-S","ShuffleNetV2","SqueezeNet1.1"]; o=A[A.cond=="orig"]
plt.rcParams.update({"font.size":7}); mk=["o","s","^","D","v","P"]
fig,ax=plt.subplots(figsize=(3.5,2.9))
for m,k in zip(M,mk):
    v=o[o.model==m].pooled; x=eff.loc[m,"lat_ms_1t_median"]
    ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt=k,capsize=2,ms=4.5,lw=1,label=f"{m.replace('1.1',' 1.1')} ({eff.loc[m,'GFLOPs']:.2f} G)")
ax.axhline(.5,ls=":",c="gray",lw=1); ax.set_ylim(.4,.97); ax.set_xlim(10,85)
ax.set_xlabel("CPU latency, 1 thread (ms, median)"); ax.set_ylabel("External AUC (Shenzhen + Montgomery)"); ax.grid(alpha=.3)
ax.legend(fontsize=5.5,loc="upper right",frameon=True,title="Model (GFLOPs)",title_fontsize=6)
plt.tight_layout(); plt.savefig("fig2_tradeoff_10seeds_legend.png",dpi=300); print("saved fig2")
