"""SV-ball optimality tests, including independent epigraph references."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.special import expit
from hapc import hapc_core as native
from hapc.core import pcghal
from hapc.single import single_pcghal, single_pcghal_classification, _calibrate_logistic_intercept
from hapc.cv import pcghal_cv, pcghal_cv_classi


def reference(y, Z, V, a0, logistic):
    n,k=Z.shape; q=V.shape[0]; C=np.linalg.norm(V@a0,1)
    dim=k+int(logistic)+q
    A=np.zeros((2*q+1,dim))
    A[:q,:k]=V;A[:q,-q:]=-np.eye(q)
    A[q:2*q,:k]=-V;A[q:2*q,-q:]=-np.eye(q)
    A[-1,-q:]=1
    upper=np.r_[np.zeros(2*q),C]
    initial=np.r_[a0, [_calibrate_logistic_intercept((y+1)/2,Z@a0)] if logistic else [],abs(V@a0)]
    def obj(x):
        eta=Z@x[:k]
        if logistic: return np.logaddexp(0,-y*(eta+x[k])).mean()
        return np.mean((y-eta)**2)
    def grad(x):
        eta=Z@x[:k]
        out=np.zeros(dim)
        if logistic:
            residual=expit(eta+x[k])-(y+1)/2
            out[:k]=Z.T@residual/n;out[k]=residual.mean()
        else: out[:k]=2*Z.T@(eta-y)/n
        return out
    fit=minimize(obj,initial,jac=grad,method="SLSQP",constraints=[LinearConstraint(A,-np.inf,upper)],
                 bounds=Bounds(np.r_[np.repeat(-np.inf,k+int(logistic)),np.zeros(q)],np.inf),
                 options=dict(ftol=1e-12,maxiter=2000))
    if not fit.success: raise AssertionError(fit.message)
    if np.linalg.norm(V@fit.x[:k],1)>C+1e-7:raise AssertionError("infeasible independent reference")
    return fit.fun,fit.x[:k]


class TestSVBall(unittest.TestCase):
    def check_certificate(self,o,V,tol):
        self.assertTrue(o.converged,(o.status,o.optimality_gap,o.dual_residual,o.iter))
        self.assertEqual(o.status,"optimal");self.assertEqual(o.constraint,"ball")
        self.assertLessEqual(np.linalg.norm(V@o.alpha,1),o.sv_radius+1e-12*(1+o.sv_radius))
        self.assertLessEqual(o.optimality_gap,tol*(1+abs(o.risk)))
        self.assertLessEqual(o.dual_bound,o.risk+1e-12)
        self.assertLess(o.dual_residual,1e-9)
        radii=np.abs(o.alphaiters@V.T).sum(1)
        self.assertTrue(np.all(radii<=o.sv_radius+1e-12*(1+o.sv_radius)))
        assert_allclose(o.alphaiters[-1],o.alpha)

    def test_zero_coordinate_is_activated(self):
        y=np.array([0.,2.]);I=np.eye(2);a=np.array([1.,0.])
        old=native.pcghal_boundary_call(y,I,I,a,500,1e-10,.8,False,"grad")
        fit=pcghal(y,I,I,a,max_iter=500,tol=1e-10)
        assert_allclose(fit.alpha,[0,1],atol=1e-7)
        self.assertAlmostEqual(fit.risk,.5);self.assertAlmostEqual(old.risk,2.5)
        self.check_certificate(fit,I,1e-10)

    def test_interior_solution_and_known_boundary(self):
        for y,a,expected in [(np.array([.2,-.1]),np.ones(2)*2,np.array([.2,-.1])),
                              (np.array([2.,1.]),np.ones(2),np.array([1.5,.5]))]:
            fit=native.pcghal_call(y,np.eye(2),np.eye(2),a,5000,1e-10,.8,False,"risk")
            self.check_certificate(fit,np.eye(2),1e-10)
            assert_allclose(fit.alpha,expected,atol=2e-5)
        self.assertLess(np.linalg.norm(native.pcghal_call(np.array([.2,-.1]),np.eye(2),np.eye(2),np.ones(2)*2).beta,1),4)

    def test_independent_epigraph_gaussian_and_logistic(self):
        for logistic in (False,True):
            for seed in range(5):
                rng=np.random.default_rng(900+seed);Z=rng.normal(size=(35,4))
                V=np.linalg.qr(rng.normal(size=(9,4)))[0]
                a=rng.normal(size=4);a/=np.linalg.norm(V@a,1)
                y=rng.choice([-1.,1.],35) if logistic else rng.normal(size=35)*3
                fn=native.pcghal_classi_call if logistic else native.pcghal_call
                args=[y,Z,V,a,15000,1e-9,.8,False]+([] if logistic else ["grad"])
                fit=fn(*args);ref,beta=reference(y,Z,V,a,logistic)
                self.check_certificate(fit,V,1e-9)
                self.assertLessEqual(fit.dual_bound,ref+1e-8)
                self.assertLessEqual(abs(fit.risk-ref),2e-7)

    def test_invariance_to_initial_direction_same_budget(self):
        rng=np.random.default_rng(22);V=np.linalg.qr(rng.normal(size=(10,4)))[0];Z=rng.normal(size=(50,4));y=rng.normal(size=50)
        fits=[]
        for a in (np.array([1.,0,0,0]),np.array([-1.,2,1,3])):
            a=a/np.linalg.norm(V@a,1)
            f=native.pcghal_call(y,Z,V,a,10000,1e-10,.8,False,"grad");self.check_certificate(f,V,1e-10);fits.append(f)
        self.assertLess(abs(fits[0].risk-fits[1].risk),1e-8)
        assert_allclose(Z@fits[0].alpha,Z@fits[1].alpha,atol=5e-5)

    def test_zero_radius_rank_deficient_loss_and_unfinished_status(self):
        f=pcghal(np.ones(2),np.eye(2),np.eye(2),np.zeros(2),tol=1e-9)
        self.check_certificate(f,np.eye(2),1e-9)
        Z=np.array([[1.,1.],[2.,2.],[3.,3.]])
        f=pcghal(np.array([2.,4.,6.]),Z,np.eye(2),np.ones(2)/2,tol=1e-9)
        self.check_certificate(f,np.eye(2),1e-9)
        f=pcghal(np.array([0.,2.]),np.eye(2),np.eye(2),np.array([1.,0.]),max_iter=0,tol=1e-10)
        self.assertFalse(f.converged);self.assertEqual(f.status,"max_iter");self.assertGreater(f.optimality_gap,0)

    def test_rank_deficient_penalty_does_not_claim_certificate(self):
        f=pcghal(np.ones(2),np.eye(2),np.array([[1.,0.],[0.,0.]]),np.ones(2),max_iter=20)
        self.assertFalse(f.converged)
        self.assertTrue(np.isinf(f.optimality_gap))

    def test_extreme_offsets_and_one_class(self):
        Z=np.c_[np.array([-1000.,-999.,800.,900.]),np.zeros(4)]
        for y in (np.array([-1.,1.,-1.,-1.]),np.ones(4),-np.ones(4)):
            f=native.pcghal_classi_call(y,Z,np.eye(2),np.array([.01,.1]),20000,1e-8,.8,False)
            self.check_certificate(f,np.eye(2),1e-8)
            if np.all(y==y[0]): self.assertEqual(f.dual_bound,0.)

    def test_discard_roundoff_pcs_in_repeated_binary_design(self):
        X=np.repeat([0.,1.],20).reshape(-1,1)
        y=X[:,0]+np.random.default_rng(2).normal(0,.1,40)
        des=native.pchal_des(X,1,20,True)
        self.assertEqual(len(des.d),1)
        fit=single_pcghal(X,y,1,20,.01,predict=X,ini="2",tol=1e-8)
        self.assertTrue(fit.optimizer_output.converged)
        self.assertTrue(np.isfinite(fit.predictions).all())

    def test_public_binomial_and_cv_diagnostics(self):
        rng=np.random.default_rng(11);X=rng.uniform(size=(40,2));y=(X[:,0]>.5).astype(float)
        fit=single_pcghal_classification(X,y,1,8,.01,predict=X,tol=1e-7)
        self.assertTrue(fit.optimizer_output.converged)
        for binary in (False,True):
            fn=pcghal_cv_classi if binary else pcghal_cv
            yy=y if binary else np.sin(6*X[:,0])
            fit=fn(X,yy,1,8,lambdas=np.array([.01,.1]),nfolds=3,tol=1e-6,predict=X)
            self.assertEqual(len(fit.sv_fold_status),6)
            self.assertTrue(all(s=="optimal" for s in fit.sv_fold_status),fit.sv_fold_status)
            self.assertTrue(fit.optimizer_output.converged)
            self.assertTrue(np.all(np.isfinite(fit.sv_fold_gap)))


if __name__=="__main__":unittest.main()
