import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
import numpy as np, pandas as pd
from scipy.stats import rankdata
R=ROOT+"/results/phase1_pilot"; L=np.load(ROOT+"/results/labels_phase1.npz")
eff=pd.read_csv(f"{R}/efficiency.csv"); MODELS=eff.model.tolist(); SEEDS=[0,1,2]
def youden(y,p):
    o=np.argsort(-p); ys=y[o]; ps=p[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); tpr=tp/ys.sum(); fpr=fp/(len(ys)-ys.sum())
    j=tpr-fpr; i=np.argmax(j); return ps[i]
def sens_spec(y,p,t):
    pred=p>=t; return (pred[y==1].mean(), (~pred[y==0]).mean())
rng=np.random.default_rng(0); REPS=200; FR=0.30
rows=[]
for setname in ["Shenzhen","Montgomery"]:
    y=L[setname]
    for mo in MODELS:
        for s in SEEDS:
            f=np.load(f"{R}/preds/{mo.replace(' ','_')}_s{s}.npz"); p=f[setname]; thr_src=float(f["thr"])
            a=[];b=[]
            for _ in range(REPS):
                cal=np.concatenate([rng.choice(np.where(y==c)[0],int(round(FR*(y==c).sum())),replace=False) for c in (0,1)])
                te=np.setdiff1d(np.arange(len(y)),cal)
                t_loc=youden(y[cal],p[cal])
                a.append(sens_spec(y[te],p[te],thr_src)); b.append(sens_spec(y[te],p[te],t_loc))
            a=np.mean(a,0); b=np.mean(b,0)
            rows.append(dict(set=setname,model=mo,seed=s,src_sens=a[0],src_spec=a[1],loc_sens=b[0],loc_spec=b[1]))
d=pd.DataFrame(rows)
d["src_bacc"]=(d.src_sens+d.src_spec)/2; d["loc_bacc"]=(d.loc_sens+d.loc_spec)/2
g=d.groupby(["set","model"])[["src_sens","src_spec","src_bacc","loc_sens","loc_spec","loc_bacc"]].mean().round(3)
g.to_csv("recalibration.csv"); print(g.to_string())
print("\nMean balanced accuracy across models:"); print(d.groupby("set")[["src_bacc","loc_bacc"]].mean().round(3))
