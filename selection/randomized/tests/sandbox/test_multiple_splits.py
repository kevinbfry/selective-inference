from __future__ import print_function
import numpy as np

import regreg.api as rr


from ...tests.flags import SMALL_SAMPLES, SET_SEED
from selection.api import (randomization,
                           split_glm_group_lasso,
                           multiple_queries)
from ...tests.instance import logistic_instance
from ...tests.decorators import wait_for_return_value, set_sampling_params_iftrue

from ..glm import (standard_split_ci,
                   glm_nonparametric_bootstrap,
                   pairs_bootstrap_glm)

from ..M_estimator import restricted_Mest
from ..query import naive_confidence_intervals

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_multiple_splits(s=3,
                         n=300,
                         p=20,
                         signal=7,
                         rho=0.1,
                         split_frac=0.8,
                         lam_frac=0.7,
                         nsplits=4,
                         ndraw=10000, burnin=2000,
                         solve_args={'min_its':50, 'tol':1.e-10}, check_screen =True):

    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)

    nonzero = np.where(beta)[0]

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    m = int(split_frac * n)

    view = []
    for i in range(nsplits):
        view.append(split_glm_group_lasso(loss, epsilon, m, penalty))

    mv = multiple_queries(view)
    mv.solve()

    active_union = np.zeros(p, np.bool)
    for i in range(nsplits):
        active_union += view[i].selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    screen = set(nonzero).issubset(np.nonzero(active_union)[0])

    if check_screen and not screen:
        return None

    true_vec = beta[active_union]
    selected_features = np.zeros(p, np.bool)
    selected_features[active_union] = True

    unpenalized_mle = restricted_Mest(loss, selected_features)

    form_covariances = glm_nonparametric_bootstrap(n, n)
    target_info, target_observed = pairs_bootstrap_glm(loss, selected_features, inactive=None)

    cov_info = view[0].setup_sampler()
    target_cov, score_cov = form_covariances(target_info,  
                                             cross_terms=[cov_info],
                                             nsample=view[0].nboot)

    for v in view:
        v.setup_sampler()
    opt_samples = [v.sampler.sample(ndraw,
                                    burnin) for v in view]

    #### XXX TODO these only use one view!
    pivots = view[0].sampler.coefficient_pvalues(unpenalized_mle, 
                                                 target_cov, 
                                                 score_cov, 
                                                 parameter=true_vec,
                                                 sample=opt_samples[0])
    LU = view[0].sampler.confidence_intervals(unpenalized_mle, target_cov, score_cov, sample=opt_samples[0])

    LU_naive = naive_confidence_intervals(np.diag(target_cov), target_observed)

    def coverage(LU):
        L, U = LU[:,0], LU[:,1]
        covered = np.zeros(nactive)
        ci_length = np.zeros(nactive)

        for j in range(nactive):
            if check_screen:
              if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            else:
                covered[j] = None
            ci_length[j] = U[j]-L[j]
        return covered, ci_length

    covered, ci_length = coverage(LU)
    covered_naive, ci_length_naive = coverage(LU_naive)

    active_set = np.where(active_union)[0]
    active_var = np.zeros(nactive, np.bool)
    for j in range(nactive):
        active_var[j] = active_set[j] in nonzero

    return (pivots, 
            covered, 
            ci_length, 
            active_var, 
            covered_naive, 
            ci_length_naive)


