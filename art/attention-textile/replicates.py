"""Phase-change timing / final scores for every toy run -> cache/toy_replicates_summary.json.  python replicates.py"""
import numpy as np, json, os
out={}
for t in ['main','seed1','seed2','onelayer']:
    p=f'cache/toy_{t}.npz'
    if not os.path.exists(p): continue
    d=np.load(p); st=d['step']; rep=d['loss_rep_later']
    lo,hi=rep[:50].mean(),rep[-50:].mean()
    k10=int(np.argmax(rep<lo-0.1*(lo-hi))); k50=int(np.argmax(rep<(lo+hi)/2)); k90=int(np.argmax(rep<lo-0.9*(lo-hi)))
    ind=d['ind_score']; prev=d['prev_score']
    on_ind=int(st[np.argmax(ind[:,-1].max(-1)>0.2)]) if (ind[:,-1].max(-1)>0.2).any() else None
    on_prev=int(st[np.argmax(prev[:,0].max(-1)>0.2)]) if (prev[:,0].max(-1)>0.2).any() else None
    out[t]=dict(loss_rep_start=float(lo),loss_rep_end=float(hi),step10=int(st[k10]) if lo-hi>0.5 else None,step50=int(st[k50]) if lo-hi>0.5 else None,step90=int(st[k90]) if lo-hi>0.5 else None,
                ind_score_final_max=float(ind[-1].max()), ind_onset_0p2=on_ind, prev_onset_0p2=on_prev,
                icl_final=float(d['icl_score'][-20:].mean()), loss_markov_final=float(d['loss_markov'][-20:].mean()), steps=int(st[-1]))
print(json.dumps(out,indent=1)); json.dump(out,open('cache/toy_replicates_summary.json','w'),indent=1)
