import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
OUT=os.path.join(ROOT,"analysis_outputs"); os.makedirs(OUT,exist_ok=True); os.chdir(OUT)
import numpy as np, pandas as pd, itertools, json
from scipy.stats import rankdata
R=ROOT+"/results/phase1_pilot"; L=np.load(ROOT+"/results/labels_phase1.npz")
m=pd.read_csv(f"{R}/metrics_per_run.csv"); eff=pd.read_csv(f"{R}/efficiency.csv").set_index("model")
MODELS=list(eff.index); SEEDS=[0,1,2]
P={}
for mo in MODELS:
    for s in SEEDS:
        f=np.load(f"{R}/preds/{mo.replace(' ','_')}_s{s}.npz")
        P[(mo,s)]={k:f[k] for k in ["TBX11K-test","Shenzhen","Montgomery"]}; P[(mo,s)]["thr"]=float(f["thr"])
def auc(y,p):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
# evaluation sets: label vector, prediction getter, strata (for stratified bootstrap)
def get(mo,s,setname):
    if setname=="SZ+MC": return np.r_[P[(mo,s)]["Shenzhen"],P[(mo,s)]["Montgomery"]]
    return P[(mo,s)][setname]
Y={"TBX11K-test":L["TBX11K-test"],"Shenzhen":L["Shenzhen"],"Montgomery":L["Montgomery"],"SZ+MC":np.r_[L["Shenzhen"],L["Montgomery"]]}
STRATA={k:(np.r_[np.zeros(662),np.ones(138)] if k=="SZ+MC" else np.zeros(len(Y[k]))) for k in Y}
rng=np.random.default_rng(0); B=1000
def boot_idx(setname):
    y=Y[setname]; st=STRATA[setname]; idx=[]
    for g in np.unique(st):
        for c in (0,1):
            ii=np.where((st==g)&(y==c))[0]; idx.append(rng.choice(ii,len(ii)))
    return np.concatenate(idx)
out={}
for setname in ["SZ+MC","Shenzhen","Montgomery","TBX11K-test"]:
    y=Y[setname]; pr={(mo,s):get(mo,s,setname) for mo in MODELS for s in SEEDS}
    point={mo:np.mean([auc(y,pr[(mo,s)]) for s in SEEDS]) for mo in MODELS}
    bs={mo:np.zeros(B) for mo in MODELS}
    for b in range(B):
        ix=boot_idx(setname); yb=y[ix]
        for mo in MODELS: bs[mo][b]=np.mean([auc(yb,pr[(mo,s)][ix]) for s in SEEDS])
    out[setname]=(point,bs)
    # seed-ensemble AUC
    ens={mo:auc(y,np.mean([pr[(mo,s)] for s in SEEDS],0)) for mo in MODELS}
    print(f"\n=== {setname}: seed-mean AUC [95% bootstrap CI over test images] | seed-ensemble AUC ===")
    for mo in MODELS:
        lo,hi=np.percentile(bs[mo],[2.5,97.5]); print(f"{mo:20s} {point[mo]:.3f} [{lo:.3f},{hi:.3f}] | ens {ens[mo]:.3f}")
    # pairwise vs SqueezeNet and Holm over all pairs
    pairs=list(itertools.combinations(MODELS,2)); rows=[]
    for a,b2 in pairs:
        d=bs[a]-bs[b2]; p=2*min((d<=0).mean(),(d>=0).mean()); p=max(p,1/B)
        rows.append((a,b2,point[a]-point[b2],*np.percentile(d,[2.5,97.5]),p))
    df=pd.DataFrame(rows,columns=["A","B","diff","lo","hi","p"]).sort_values("p")
    k=len(df); df["p_holm"]=[min(1,(k-i)*p) for i,p in enumerate(df.p)]; df["p_holm"]=np.maximum.accumulate(df.p_holm)
    df.to_csv(f"pairwise_{setname.replace('+','_')}.csv",index=False)
    sig=df[df.p_holm<0.05]
    print(f"significant pairs after Holm (alpha .05): {len(sig)} of {k}")
    print(sig.round(3).to_string(index=False) if len(sig) else "  none")
# seed variance vs model variance (pooled external)
rows=[]
for setname in ["SZ+MC","Shenzhen","Montgomery","TBX11K-test"]:
    y=Y[setname]; a=pd.DataFrame([(mo,s,auc(y,get(mo,s,setname))) for mo in MODELS for s in SEEDS],columns=["model","seed","auc"])
    within=a.groupby("model").auc.var(ddof=1).mean(); between=a.groupby("model").auc.mean().var(ddof=1)
    rows.append((setname,within**.5,between**.5,(a.auc.max()-a.auc.min())))
print("\n=== seed-to-seed SD (avg within model) vs between-model SD of mean AUC, and overall range ===")
print(pd.DataFrame(rows,columns=["set","within_seed_SD","between_model_SD","range"]).round(3).to_string(index=False))
import pickle; pickle.dump({k:(v[0],{m:b for m,b in v[1].items()}) for k,v in out.items()},open("boot.pkl","wb"))
