from __future__ import print_function

import numpy as np
import regreg.api as rr
import nose.tools as nt

try:
    import rpy2.robjects as rpy
    rpy2_available = True
    import rpy2.robjects.numpy2ri as numpy2ri
except ImportError:
    rpy2_available = False

from ..lasso import lasso
from ...tests.instance import gaussian_instance, logistic_instance

@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_randomized_lasso(n=100, p=20, s=4, rho=0.4):
    """
    Check that Gaussian randomized LASSO results agree with R
    """
    numpy2ri.activate()

    tol = 1.e-5
    for s in [1,1.1]:
        lam = 7.8

        rpy.r.assign('n', n)
        rpy.r.assign('p', p)
        rpy.r.assign('s', s)
        rpy.r.assign('rho', rho)

        R_code = """

        snr = sqrt(2*log(p)/n)

        set.seed(1)
        construct_ci=TRUE
        penalty_factor = rep(1, p)

        data = selectiveInference:::gaussian_instance(n=n, p=p, s=s, rho=rho, sigma=1, snr=snr)

        X=data$X
        y=data$y
        beta=data$beta

        sigma_est=1

        lambda = 0.7 * selectiveInference:::theoretical.lambda(X, "ls", sigma_est)  # theoretical lambda

        rand_lasso_soln = selectiveInference:::randomizedLasso(X, 
                                                               y, 
                                                               lambda*n, 
                                                               family=selectiveInference:::family_label("ls"))

         targets=selectiveInference:::compute_target(rand_lasso_soln, type="selected", sigma_est=sigma_est)

         inf_obj = selectiveInference:::randomizedLassoInf(rand_lasso_soln,
                                                           targets=targets,
                                                           sampler = "norejection", #"adaptMCMC", #
                                                           level=0.9, 
                                                           burnin=1000, 
                                                           nsample=10000)
         active_vars=rand_lasso_soln$active_set

        """ 

        rpy.r(R_code)

        inf_obj = rpy.r('inf_obj')
        rand_lasso_obj = rpy.r('rand_lasso_soln')
        stop
        numpy2ri.deactivate()

#         yield np.testing.assert_allclose, L.fit()[1:], beta_hat, 1.e-2, 1.e-2, False, 'fixed lambda, sigma=%f coef' % s
#         yield np.testing.assert_equal, L.active, selected_vars
#         yield np.testing.assert_allclose, S['pval'], R_pvals, tol, tol, False, 'fixed lambda, sigma=%f pval' % s
#         yield np.testing.assert_allclose, S['sd'], sdvar, tol, tol, False, 'fixed lambda, sigma=%f sd ' % s
#         yield np.testing.assert_allclose, S['onestep'], coef, tol, tol, False, 'fixed lambda, sigma=%f estimator' % s

