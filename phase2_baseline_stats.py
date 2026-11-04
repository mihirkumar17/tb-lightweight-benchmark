import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
import numpy as np, pandas as pd, glob, itertools
from scipy.stats import rankdata, kruskal, mannwhitneyu, wilcoxon
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
def auc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
MODELS=["MobileNetV3-L","EfficientNet-B0","EfficientNet-Lite0","MobileViT-S","ShuffleNetV2","SqueezeNet1.1"]
rows=[]
for f in sorted(glob.glob(ROOT+"/results/phase2_commit2_base_10seeds/preds/base_*.npz")):
    z=np.load(f); nm=f.split("tbx_")[1][:-4]; mo,seed=nm.rsplit("_s",1); seed=int(seed)
    y_ext=np.r_[z["Shenzhen|y"],z["Montgomery|y"]]
    for cond in ["orig","border","corners","border+corners","centercrop80"]:
        p_ext=np.r_[z[f"Shenzhen|{cond}"],z[f"Montgomery|{cond}"]]
        thr=float(z["thr"])
        rows.append(dict(model=mo,seed=seed,cond=cond,
            tbx=auc(z["TBX11K-test|y"],z[f"TBX11K-test|{cond}"]),sz=auc(z["Shenzhen|y"],z[f"Shenzhen|{cond}"]),
            mc=auc(z["Montgomery|y"],z[f"Montgomery|{cond}"]),pooled=auc(y_ext,p_ext),
            fl_sz=(z[f"Shenzhen|{cond}"]>=thr).mean(),fl_mc=(z[f"Montgomery|{cond}"]>=thr).mean(),fl_tbx=(z[f"TBX11K-test|{cond}"]>=thr).mean(),thr=thr))
A=pd.DataFrame(rows); A.to_csv("per_run_auc.csv",index=False)
o=A[A.cond=="orig"]
def ms(s): return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
print("=== 10-SEED BASELINE (original images): AUC mean ± sample SD [min, max] ===")
for k,lab in [("tbx","TBX11K-test"),("sz","Shenzhen"),("mc","Montgomery"),("pooled","SZ+MC pooled")]:
    print(f"\n{lab}")
    for mo in MODELS:
        v=o[o.model==mo][k]; print(f"  {mo:20s} {ms(v)} [{v.min():.3f}, {v.max():.3f}]")
print("\n=== Seed variance vs model variance ===")
for k in ["tbx","sz","mc","pooled"]:
    within=o.groupby("model")[k].var(ddof=1).mean()**.5; between=o.groupby("model")[k].mean().std(ddof=1)
    H,p=kruskal(*[o[o.model==m][k].values for m in MODELS])
    print(f"{k:7s} within-seed SD={within:.3f} between-model SD of means={between:.3f} range={o[k].min():.3f}-{o[k].max():.3f} | Kruskal-Wallis p={p:.4f}")
print("\n=== Pairwise Mann-Whitney on pooled external AUC (10 vs 10 seeds), Holm ===")
pr=[]
for a,b in itertools.combinations(MODELS,2):
    u=mannwhitneyu(o[o.model==a].pooled,o[o.model==b].pooled,alternative="two-sided"); pr.append((a,b,o[o.model==a].pooled.mean()-o[o.model==b].pooled.mean(),u.pvalue))
P=pd.DataFrame(pr,columns=["A","B","diff","p"]).sort_values("p"); k=len(P)
P["p_holm"]=np.maximum.accumulate([min(1,(k-i)*p) for i,p in enumerate(P.p)]); print(P.round(4).head(8).to_string(index=False)); print("significant after Holm:",int((P.p_holm<.05).sum()),"of",k)
P.to_csv("pairwise_seed_level.csv",index=False)
# same for Shenzhen and Montgomery
for kk in ["sz","mc"]:
    pr=[]
    for a,b in itertools.combinations(MODELS,2):
        u=mannwhitneyu(o[o.model==a][kk],o[o.model==b][kk],alternative="two-sided"); pr.append(u.pvalue)
    pr=np.sort(pr); h=np.maximum.accumulate([min(1,(len(pr)-i)*p) for i,p in enumerate(pr)]); print(kk,"pairs significant after Holm:",int((h<.05).sum()),"of",len(pr))
print("\n=== MASKING: mean change in AUC vs original (paired by seed), with sign-test counts ===")
wide=A.pivot_table(index=["model","seed"],columns="cond",values=["tbx","sz","mc","pooled"])
out=[]
for k,lab in [("tbx","TBX11K-test"),("sz","Shenzhen"),("mc","Montgomery"),("pooled","SZ+MC pooled")]:
    for c in ["border","corners","border+corners","centercrop80"]:
        d=(wide[(k,c)]-wide[(k,"orig")]).unstack(0)
        row={"set":lab,"cond":c}
        for mo in MODELS: row[mo]=f"{d[mo].mean():+.3f}"
        row["ALL mean"]=f"{d.values.mean():+.3f}"; 
        try: row["Wilcoxon p (all 60 pairs)"]=f"{wilcoxon(d.values.ravel()).pvalue:.4f}"
        except Exception: row["Wilcoxon p (all 60 pairs)"]="n/a"
        out.append(row)
print(pd.DataFrame(out).to_string(index=False))
print("\n=== MASKING: flagged fraction (mean over models & seeds) ===")
print(A.groupby("cond")[["fl_tbx","fl_sz","fl_mc"]].mean().round(3).to_string())
print("\n=== Youden threshold across seeds: min/median/max ==="); print(o.groupby("model").thr.agg(["min","median","max"]).round(3).to_string())
