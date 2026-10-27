"""Compare HEAD-source baseline against working-source SV kernels in isolation.
Needs clang++, Eigen headers, numpy. Does not change installed hapc libraries.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-ref", default="84bdb8ac54f0858965c29d6bb9d17c571500fe8a")
parser.add_argument("--eigen-include", default="/opt/homebrew/include/eigen3")
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
args.output.mkdir(parents=True,exist_ok=True)
build=Path(tempfile.mkdtemp(prefix="hapc-sv-"))
for version in ("baseline", "revised"):
    source=build/version;source.mkdir()
    files=["hapc_core.hpp","pcghal_call.cpp","pcghal_classi_call.cpp"]
    if version=="revised": files += ["sv_descent.hpp","logistic_intercept.hpp","sv_ball.hpp"]
    for name in files:
        data=(subprocess.check_output(["git","show",f"{args.baseline_ref}:src/{name}"],cwd=root)
              if version=="baseline" else (root/"src"/name).read_bytes())
        (source/name).write_bytes(data)
    command=["clang++","-O2","-std=c++17","-shared","-fPIC",
             "-I"+args.eigen_include,"-I"+str(source)]
    if version=="revised":command += ["-DREPORT_STATUS"]
    command += [str(Path(__file__).with_name("native_bridge.cpp")),
                str(source/"pcghal_call.cpp"),str(source/"pcghal_classi_call.cpp"),
                "-o",str(build/(version+".dylib"))]
    subprocess.run(command,check=True)

from pathlib import Path
import json
import numpy as np
from native_bridge import fit
rows=[]
for logistic in (0,1):
 for seed in range(20):
  rng=np.random.default_rng(seed);V=np.linalg.qr(rng.normal(size=(9,4)))[0];Z=rng.normal(size=(30,4))*10;a=rng.normal(size=4)
  a[-1]=-(V[0,:-1]@a[:-1])/V[0,-1]
  y=rng.choice([-1.,1.],30) if logistic else rng.normal(size=30)*100
  for name in ('baseline','revised'):
   h,r,s=fit(build/(name+'.dylib'),y,Z,V,a,logistic,250)
   norms=abs(h@V.T).sum(1); inc=None
   if not logistic:inc=float(np.max(np.diff(np.mean((y[None,:]-h@Z.T)**2,axis=1)),initial=0))
   rows.append(dict(family='binomial' if logistic else 'gaussian',seed=seed,version=name,max_relative_sv_drift=float(np.max(abs(norms/norms[0]-1))),max_risk_increase=inc,status=s,iterations=len(h)-1))
p=args.output/'baseline_comparison.json';p.write_text(json.dumps(rows,indent=2)+'\n')
for fam in ('gaussian','binomial'):
 for name in ('baseline','revised'):
  rr=[r for r in rows if r['family']==fam and r['version']==name]
  print(fam,name,'maxdrift',max(r['max_relative_sv_drift'] for r in rr),'casesdrift',sum(r['max_relative_sv_drift']>1e-8 for r in rr),'maxriskincrease',max(r['max_risk_increase'] or 0 for r in rr))
