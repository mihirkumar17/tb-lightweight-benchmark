"""LODO table, masked-training mitigation, per-seed border dependence, flagged fractions.
Run after phase2_baseline_stats.py (needs per_run_auc.csv in analysis_outputs/)."""
import os, glob, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
ms=lambda s: f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"
C1=ROOT+"/results/phase2_commit1_lodo_mitigation"
M=pd.concat([pd.read_csv(f) for f in glob.glob(C1+"/*.csv")],ignore_index=True) if False else pd.read_csv(C1+"/all_metrics.csv")
# ---------------- LODO ----------------
lo=M[M.plan=="lodo"]
t=lo.groupby(["tag","model"]).auc.agg(ms).unstack(0); t.to_csv("lodo_table.csv"); print("=== LODO AUC (3 seeds) ===\n",t.to_string(),"\n")
print("mean AUC over the 4 models per fold:\n",lo.groupby(["tag","model"]).auc.mean().groupby("tag").mean().round(3).to_string(),"\n")
print("flagged fraction vs prevalence:\n",lo.groupby("tag")[["flagged","prevalence"]].mean().round(3).to_string(),"\n")
# ---------------- mitigation vs 10-seed baseline ----------------
A=pd.read_csv("per_run_auc.csv"); base=A[A.cond=="orig"]
rows=[]
def auc(y,p): return roc_auc_score(y,p)
for f in sorted(glob.glob(C1+"/preds/mitig_*.npz")):
    z=np.load(f); nm=os.path.basename(f)[len("mitig_tbx_"):-4]; mo,seed=nm.rsplit("_s",1)
    for cond in ["border+corners","orig"]:
        y=np.r_[z["Shenzhen|y"],z["Montgomery|y"]]; p=np.r_[z[f"Shenzhen|{cond}"],z[f"Montgomery|{cond}"]]
        rows.append(dict(model=mo.strip("_"),seed=int(seed),cond=cond,pooled=auc(y,p),sz=auc(z["Shenzhen|y"],z[f"Shenzhen|{cond}"]),
                         mc=auc(z["Montgomery|y"],z[f"Montgomery|{cond}"]),tbx=auc(z["TBX11K-test|y"],z[f"TBX11K-test|{cond}"])))
R=pd.DataFrame(rows); R.to_csv("mitigation_per_run.csv",index=False)
out=[]
for mo in ["MobileNetV3-L","ShuffleNetV2","SqueezeNet1.1"]:
    b=base[base.model==mo]
    for cond,lab in [("border+corners","masked-train/masked-test"),("orig","masked-train/original-test")]:
        r=R[(R.model==mo)&(R.cond==cond)]
        d=dict(model=mo,setting=lab,n_seeds=len(r),baseline_pooled=f"{b.pooled.mean():.3f} ± {b.pooled.std(ddof=1):.3f}",masked_pooled=f"{r.pooled.mean():.3f} ± {r.pooled.std(ddof=1):.3f}")
        d["p_pooled_exact_MW"]=round(mannwhitneyu(r.pooled,b.pooled,alternative="two-sided",method="exact").pvalue,3)
        d["internal_masked_vs_base"]=f"{r.tbx.mean():.3f} vs {b.tbx.mean():.3f}"; d["p_internal"]=round(mannwhitneyu(r.tbx,b.tbx,alternative="two-sided",method="exact").pvalue,3)
        out.append(d)
T=pd.DataFrame(out); T.to_csv("mitigation_table.csv",index=False); print("=== MITIGATION vs 10-seed baseline (exploratory, unadjusted; Bonferroni threshold 0.05/3=0.017) ===\n",T.to_string(index=False),"\n")
# ---------------- border dependence per seed ----------------
w=A.pivot_table(index=["model","seed"],columns="cond",values="tbx"); drop=(w["orig"]-w["border"]).unstack(0)
dep=pd.DataFrame({"mean_internal_auc_border_masked":w["border"].unstack(0).mean(),"sd":w["border"].unstack(0).std(ddof=1),"runs_drop_gt_0.05":(drop>0.05).sum()})
dep.to_csv("border_dependence_internal.csv"); print("=== Internal AUC drop under border masking ===\n",dep.round(3).to_string(),"\n")
o=base.set_index(["model","seed"]).join((w["orig"]-w["border"]).rename("border_dep"))
print(f"corr(border dependence, pooled external AUC) over {len(o)} runs: Pearson={o.border_dep.corr(o.pooled):.3f}, Spearman={o.border_dep.corr(o.pooled,method='spearman'):.3f}\n")
fl=A.groupby("cond")[["fl_tbx","fl_sz","fl_mc"]].mean().round(3); fl.to_csv("flagged_by_condition.csv"); print("=== flagged fraction by image condition ===\n",fl.to_string())
