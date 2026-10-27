"""Matched-radius comparison: boundary descent vs convex SV ball.
Includes all cases, statuses, certificates and independent reference fits.
"""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
from hapc import hapc_core as core


def row(family,case,method,o,V,elapsed):
    return dict(family=family,case=case,method=method,radius=o.sv_radius,
        coefficient_sv=np.linalg.norm(V@o.alpha,1),risk=o.risk,iterations=o.iter,
        status=o.status,converged=o.converged,optimality_gap=o.optimality_gap,
        dual_bound=o.dual_bound,dual_residual=o.dual_residual,seconds=elapsed)


def run(output):
    output.mkdir(parents=True,exist_ok=True);rows=[];references=[]
    for n in (32,64,128,256):
        rng=np.random.default_rng(20261027+n)
        x=(np.arange(n)+.5)/n;knots=(x[:-1]+x[1:])/2
        H=(x[:,None]>=knots).astype(float);H-=H.mean(0)
        U,D,Vt=np.linalg.svd(H,full_matrices=False)
        U,D,V=U[:,:12],D[:12],Vt[:12].T;Z=U*D
        y=Z@np.r_[2.,np.repeat(.02,11)]+rng.normal(0,.03,n);y-=y.mean()
        def ridge(lam):return core.ridge_call(y,U,D**2,lam)
        lo,hi=0.,1.
        while np.linalg.norm(V@ridge(hi),1)>1:hi*=2
        for _ in range(90):
            mid=(lo+hi)/2
            if np.linalg.norm(V@ridge(mid),1)>1:lo=mid
            else:hi=mid
        initial=ridge(hi)
        for method,fn in (("boundary",core.pcghal_boundary_call),("ball",core.pcghal_call)):
            t=time.perf_counter();o=fn(y,Z,V,initial,10000,1e-8,.8,False,"grad")
            rows.append(row("HAL_gaussian",str(n),method,o,V,time.perf_counter()-t))
    for logistic in (False,True):
        for seed in range(20):
            rng=np.random.default_rng(seed);V=np.linalg.qr(rng.normal(size=(9,4)))[0]
            Z=rng.normal(size=(30,4))*10;initial=rng.normal(size=4)
            initial[-1]=-(V[0,:-1]@initial[:-1])/V[0,-1]
            y=rng.choice([-1.,1.],30) if logistic else rng.normal(size=30)*100
            for method in ("boundary","ball"):
                fn=(core.pcghal_boundary_classi_call if method=="boundary" else core.pcghal_classi_call) if logistic else (core.pcghal_boundary_call if method=="boundary" else core.pcghal_call)
                t=time.perf_counter();o=fn(*([y,Z,V,initial,10000,1e-8,.8,False]+([] if logistic else ["grad"])))
                rows.append(row("binomial" if logistic else "gaussian",str(seed),method,o,V,time.perf_counter()-t))
    # The independent SLSQP epigraph uses full alpha/intercept/t variables,
    # not the ADMM update or its stopping rule. Failure is fatal, not excluded.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tests"))
    from test_sv_ball import reference
    for logistic in (False,True):
        for seed in range(5):
            rng=np.random.default_rng(900+seed);Z=rng.normal(size=(35,4));V=np.linalg.qr(rng.normal(size=(9,4)))[0]
            initial=rng.normal(size=4);initial/=np.linalg.norm(V@initial,1)
            y=rng.choice([-1.,1.],35) if logistic else rng.normal(size=35)*3
            fn=core.pcghal_classi_call if logistic else core.pcghal_call
            o=fn(*([y,Z,V,initial,15000,1e-9,.8,False]+([] if logistic else ["grad"])))
            val,_=reference(y,Z,V,initial,logistic)
            references.append(dict(family="binomial" if logistic else "gaussian",seed=seed,
                risk=o.risk,reference_risk=val,absolute_difference=abs(o.risk-val),
                dual_bound=o.dual_bound,gap=o.optimality_gap,status=o.status))
    for filename,data in (("comparison.csv",rows),("independent_reference.csv",references)):
        with (output/filename).open("w") as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    config=dict(seed="HAL:20261027+n; stress:0..19; reference:900..904",
        halt_sizes=[32,64,128,256],stress_cases=40,reference_cases=10,
        max_iter=10000,tol=1e-8,reference_tol=1e-9,
        constraint="coefficient SV <= radius from the same initial fit for both methods",
        note="Synthetic optimisation validation, not CV calibration or a convergence-rate experiment.")
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    for family in ("HAL_gaussian","gaussian","binomial"):
        for method in ("boundary","ball"):
            rr=[r for r in rows if r["family"]==family and r["method"]==method]
            print(family,method,"cases",len(rr),"statuses",{s:sum(r["status"]==s for r in rr) for s in set(r["status"] for r in rr)},
                  "max_gap",max(r["optimality_gap"] for r in rr))
    print("reference_max_abs_difference",max(r["absolute_difference"] for r in references))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);run(p.parse_args().output)
