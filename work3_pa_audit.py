import pandas as pd, numpy as np, hashlib, json, os
from urllib.request import urlretrieve
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

OUT='audit_outputs'; os.makedirs(OUT,exist_ok=True)
URL_EC='https://raw.githubusercontent.com/RK2627/PromoDGDE/main/Data/ecoli/ecoli_exp.csv'
URL_PA='https://raw.githubusercontent.com/WangLabTHU/DeepCROSS/main/dataset/AAE_supervised/calibration_nm2018.csv'
urlretrieve(URL_EC,'repo_ec.csv'); urlretrieve(URL_PA,'nm2018.csv')

def normseq(x): return str(x).strip().upper()
def read_ec(p):
    d=pd.read_csv(p,header=None,names=['sequence','activity']); d.sequence=d.sequence.map(normseq); return d
EC=read_ec('repo_ec.csv')
PA0=pd.read_csv('nm2018.csv')
if 'seqs' not in PA0.columns: raise RuntimeError(PA0.columns.tolist())
pa_col='PA(2018)'
PA=pd.DataFrame({'sequence':PA0.seqs.map(normseq),'activity':pd.to_numeric(PA0[pa_col],errors='coerce'),'oligo_id':PA0['OLIGO ID'] if 'OLIGO ID' in PA0 else np.arange(len(PA0))})
PA=PA[PA.activity.notna() & PA.sequence.str.fullmatch('[ACGT]+') & (PA.sequence.str.len()==165)].copy()
PA=PA.drop_duplicates('sequence',keep='first').reset_index(drop=True)
EC=EC[EC.activity.notna() & EC.sequence.str.fullmatch('[ACGT]+') & (EC.sequence.str.len()==165)].drop_duplicates('sequence').reset_index(drop=True)

def sethash(seqs): return hashlib.sha256(('\n'.join(sorted(set(seqs)))+'\n').encode()).hexdigest()
def enc(seqs):
    lut=np.full(256,255,np.uint8); lut[ord('A')]=0; lut[ord('C')]=1; lut[ord('G')]=2; lut[ord('T')]=3
    b=np.frombuffer(''.join(seqs).encode(),dtype=np.uint8).reshape(len(seqs),165)
    return lut[b]
A=enc(EC.sequence.tolist()); B=enc(PA.sequence.tolist())
comp=np.array([3,2,1,0],dtype=np.uint8); Brc=comp[B[:,::-1]]
maxid=np.zeros(len(B),np.float32); orient=np.empty(len(B),dtype='U2'); bs=64
for i in range(0,len(B),bs):
    x=B[i:i+bs]; xr=Brc[i:i+bs]
    best=np.full(len(x),166,dtype=np.int16); best_o=np.array(['F']*len(x),dtype='U2')
    for j in range(0,len(A),2048):
        y=A[j:j+2048]
        d=(x[:,None,:]!=y[None,:,:]).sum(2); br=(xr[:,None,:]!=y[None,:,:]).sum(2)
        m=d.min(1); mr=br.min(1)
        use=m<best; best[use]=m[use]; best_o[use]='F'
        use=mr<best; best[use]=mr[use]; best_o[use]='RC'
    maxid[i:i+len(x)]=1-best/165.0; orient[i:i+len(x)]=best_o
PA['max_identity_to_EC']=maxid; PA['nearest_orientation']=orient
for t in [1.0,.95,.90,.85,.80,.70]: PA[f'overlap_ge_{t:.2f}']=PA.max_identity_to_EC >= t-1e-12
PA['keep_primary_lt_0.80']=PA.max_identity_to_EC < .80-1e-12
PA.to_csv(f'{OUT}/PA_Johns2018_full_audit.csv',index=False)
C=PA[PA['keep_primary_lt_0.80']].copy(); C[['sequence','activity','oligo_id']].to_csv(f'{OUT}/PA_Johns2018_nonpaired_lt80.csv',index=False)

def gc(s): return (s.count('G')+s.count('C'))/len(s)
def kmer_dist(seqs,k=6):
    v=np.zeros(4**k,dtype=np.float64); mp={'A':0,'C':1,'G':2,'T':3}
    for s in seqs:
        z=0
        for i,ch in enumerate(s):
            z=((z<<2)+mp[ch]) & ((1<<(2*k))-1)
            if i>=k-1: v[z]+=1
    return v/v.sum()
def pair_metrics(a,b):
    ga=np.array([gc(s) for s in a]); gb=np.array([gc(s) for s in b])
    p=kmer_dist(a); q=kmer_dist(b); js=float(jensenshannon(p,q,base=2)**2)
    seq=list(a)+list(b); y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    V=CountVectorizer(analyzer='char',ngram_range=(6,6),lowercase=False,dtype=np.float32); X=V.fit_transform(seq)
    cv=StratifiedKFold(5,shuffle=True,random_state=20261012)
    auc=float(cross_val_score(LogisticRegression(max_iter=1000,C=.5,n_jobs=1),X,y,cv=cv,scoring='roc_auc',n_jobs=1).mean())
    Z=TruncatedSVD(n_components=20,random_state=20261012).fit_transform(X)
    za=Z[:len(a)]; zb=Z[len(a):]; pooled=np.sqrt((za.var(0).mean()+zb.var(0).mean())/2)+1e-12
    cent=float(np.linalg.norm(za.mean(0)-zb.mean(0))/pooled)
    return dict(n_a=len(a),n_b=len(b),gc_mean_a=float(ga.mean()),gc_mean_b=float(gb.mean()),gc_abs_mean_diff=float(abs(ga.mean()-gb.mean())),gc_ks=float(ks_2samp(ga,gb).statistic),gc_wasserstein=float(wasserstein_distance(ga,gb)),jsd_6mer=js,host_auc_6mer=auc,svd20_centroid_distance_scaled=cent)
M=pair_metrics(EC.sequence.tolist(),C.sequence.tolist())
counts={f'ge_{t:.2f}':int((PA.max_identity_to_EC>=t-1e-12).sum()) for t in [1,.95,.90,.85,.80,.70]}
summary={'repo_ec_rows':int(len(EC)),'repo_ec_unique':int(EC.sequence.nunique()),'repo_ec_sequence_set_sha256':sethash(EC.sequence),'pa_raw_rows':int(len(PA0)),'pa_qc_unique_165':int(len(PA)),'similarity_counts':counts,'pa_retained_lt_0.80':int(len(C)),'pa_retention_fraction_lt_0.80':float(len(C)/len(PA)),'ec_pa_domain_metrics_after_filter':M,'sources':{'ec':URL_EC,'pa':URL_PA},'primary_similarity_rule':'exclude PA sequence if max aligned identity to any EC sequence or its reverse-complement is >=0.80'}
with open(f'{OUT}/summary.json','w') as f: json.dump(summary,f,indent=2)
pd.DataFrame([M]).to_csv(f'{OUT}/EC_PA_domain_metrics.csv',index=False)
with open(f'{OUT}/README.txt','w') as f: f.write('Work3 Branch03 EC-PA controlled non-paired audit\n'+json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
