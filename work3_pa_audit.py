import pandas as pd, numpy as np, hashlib, json, os, subprocess
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
def ns(x): return str(x).strip().upper()
EC=pd.read_csv('repo_ec.csv',header=None,names=['sequence','activity']); EC.sequence=EC.sequence.map(ns)
P0=pd.read_csv('nm2018.csv'); PA=pd.DataFrame({'sequence':P0.seqs.map(ns),'activity':pd.to_numeric(P0['PA(2018)'],errors='coerce'),'oligo_id':P0['OLIGO ID'] if 'OLIGO ID' in P0 else np.arange(len(P0))})
EC=EC[EC.activity.notna() & EC.sequence.str.fullmatch('[ACGT]+') & (EC.sequence.str.len()==165)].drop_duplicates('sequence').reset_index(drop=True)
PA=PA[PA.activity.notna() & PA.sequence.str.fullmatch('[ACGT]+') & (PA.sequence.str.len()==165)].drop_duplicates('sequence').reset_index(drop=True)
def sethash(s): return hashlib.sha256(('\n'.join(sorted(set(s)))+'\n').encode()).hexdigest()
with open('ec.fa','w') as f:
    for i,s in enumerate(EC.sequence): f.write(f'>e{i}\n{s}\n')
with open('pa.fa','w') as f:
    for i,s in enumerate(PA.sequence): f.write(f'>p{i}\n{s}\n')
# Global-alignment identity; both strands; report top hit >=70%.
cmd=['vsearch','--usearch_global','pa.fa','--db','ec.fa','--id','0.70','--strand','both','--maxaccepts','0','--maxrejects','0','--maxhits','1','--userout','hits.tsv','--userfields','query+target+id','--threads','0']
subprocess.run(cmd,check=True)
mx=np.zeros(len(PA),dtype=float); targ=np.array(['']*len(PA),dtype=object)
if os.path.getsize('hits.tsv'):
    H=pd.read_csv('hits.tsv',sep='\t',header=None,names=['q','t','id'])
    for r in H.itertuples(index=False):
        i=int(str(r.q)[1:]); mx[i]=float(r.id)/100; targ[i]=r.t
PA['max_global_identity_to_EC']=mx; PA['nearest_EC_id']=targ
for t in [1,.95,.90,.85,.80,.70]: PA[f'overlap_ge_{t:.2f}']=PA.max_global_identity_to_EC >= t-1e-12
PA['keep_primary_lt_0.80']=PA.max_global_identity_to_EC < .80-1e-12
PA.to_csv(f'{OUT}/PA_Johns2018_full_audit.csv',index=False)
C=PA[PA['keep_primary_lt_0.80']].copy(); C[['sequence','activity','oligo_id']].to_csv(f'{OUT}/PA_Johns2018_nonpaired_lt80.csv',index=False)
def gc(s): return (s.count('G')+s.count('C'))/len(s)
def kd(seqs,k=6):
    v=np.zeros(4**k,float); mp={'A':0,'C':1,'G':2,'T':3}; mask=(1<<(2*k))-1
    for s in seqs:
        z=0
        for i,ch in enumerate(s):
            z=((z<<2)+mp[ch]) & mask
            if i>=k-1: v[z]+=1
    return v/v.sum()
def metrics(a,b):
    ga=np.array([gc(s) for s in a]); gb=np.array([gc(s) for s in b]); p=kd(a); q=kd(b)
    seq=list(a)+list(b); y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    X=CountVectorizer(analyzer='char',ngram_range=(6,6),lowercase=False,dtype=np.float32).fit_transform(seq)
    cv=StratifiedKFold(5,shuffle=True,random_state=20261012)
    auc=float(cross_val_score(LogisticRegression(max_iter=1000,C=.5),X,y,cv=cv,scoring='roc_auc',n_jobs=-1).mean())
    Z=TruncatedSVD(n_components=20,random_state=20261012).fit_transform(X); za=Z[:len(a)]; zb=Z[len(a):]
    pooled=np.sqrt((za.var(0).mean()+zb.var(0).mean())/2)+1e-12
    return {'n_a':len(a),'n_b':len(b),'gc_mean_a':float(ga.mean()),'gc_mean_b':float(gb.mean()),'gc_abs_mean_diff':float(abs(ga.mean()-gb.mean())),'gc_ks':float(ks_2samp(ga,gb).statistic),'gc_wasserstein':float(wasserstein_distance(ga,gb)),'jsd_6mer':float(jensenshannon(p,q,base=2)**2),'host_auc_6mer':auc,'svd20_centroid_distance_scaled':float(np.linalg.norm(za.mean(0)-zb.mean(0))/pooled)}
M=metrics(EC.sequence.tolist(),C.sequence.tolist()); counts={f'ge_{t:.2f}':int((PA.max_global_identity_to_EC>=t-1e-12).sum()) for t in [1,.95,.90,.85,.80,.70]}
S={'repo_ec_rows':len(EC),'repo_ec_unique':int(EC.sequence.nunique()),'repo_ec_sequence_set_sha256':sethash(EC.sequence),'pa_raw_rows':len(P0),'pa_qc_unique_165':len(PA),'similarity_counts':counts,'pa_retained_lt_0.80':len(C),'pa_retention_fraction_lt_0.80':len(C)/len(PA),'ec_pa_domain_metrics_after_filter':M,'sources':{'ec':URL_EC,'pa':URL_PA},'primary_similarity_rule':'exclude PA if best 165-bp global-alignment identity to EC (either strand) is >=0.80'}
with open(f'{OUT}/summary.json','w') as f: json.dump(S,f,indent=2)
pd.DataFrame([M]).to_csv(f'{OUT}/EC_PA_domain_metrics.csv',index=False)
with open(f'{OUT}/README.txt','w') as f: f.write('Work3 Branch03 EC-PA controlled non-paired audit\n'+json.dumps(S,indent=2))
print(json.dumps(S,indent=2))
