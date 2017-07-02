from datetime import datetime
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def mapping(data, theta_start, update_w, update_change, regularizer,
            mapping_iter, tol):
    """produces the regularized estimate for the covariance using the
    proximal gradient procedure described in

    http://dept.stat.lsa.umich.edu/~yvesa/sto_prox.pdf

    Parameters
    ----------

    data : array-like
        N x P matrix of data for corresponding subset
    theta_start : array-like
        P x P matrix, starting value for theta estimation
    update_w : scalar
        weights applied to update theta estimate, gamma in the
        source paper
    update_change : scalar
        chagne to update_w when estimation procedure fails. New
        update_w *= update_change
    regularizer : scalar
        Regularizing constant, lamb in source paper
    mapping_iter ; scalar
        maximum number of iterations allowed for the mapping
    tol : scalar
        tolerance at which the mapping is stopped if before
        mapping iter

    Returns
    -------
    P x P matrix corresponding to theta estimate
    """

    # TODO this currently only supports LASSO not elastic net, fairly trivial
    # to modify but hasn't been done yet.

    n, p = data.shape
    cov_est = np.cov(data.T)

    theta_p = np.eye(p)
    theta_k = theta_start.copy()

    # update the regularizer to reflect data subset
    regularizer *= np.sqrt(np.log(p) / n)

    i = 0

    norm = np.linalg.norm(theta_k - theta_p) / np.linalg.norm(theta_k)

    state = True
    while state and i < mapping_iter:

        try:
            inv_theta = np.linalg.inv(theta_k)
            theta_p = theta_k - update_w * (cov_est - inv_theta)
            thresh0 = np.abs(theta_p) < regularizer * update_w
            thresh1 = theta_p >= regularizer * update_w
            thresh2 = theta_p <= -regularizer * update_w
            theta_p[thresh0] = 0
            theta_p[thresh1] = theta_p[thresh1] - regularizer * update_w
            theta_p[thresh2] = theta_p[thresh2] + regularizer * update_w
            norm = np.linalg.norm(theta_k - theta_p) / np.linalg.norm(theta_k)
            theta_k = theta_p
            i += 1

        except Exception as e:
            update_w *= update_change
            theta_k = theta_start.copy()
            print e, update_w, np.linalg.norm(theta_p), np.sum(theta_p == 0), regularizer * update_w
        if norm < tol:
            state = False

    return theta_k


def kernel_gaussian(tau, n, buff, sigma2):
    """samples neighbors with gaussian kernel

    Parameters
    ----------

    tau : scalar
        current changepoint
    n : scalar
        sample size
    buff : scalar
        buffer to avoid edges
    sigma2 : scalar
        proposal variance

    Return
    ------
    proposal for changepoint
    """

    T = np.delete(np.arange(1, n), tau - 1)
    T = np.delete(T, range(0, buff))
    T = np.delete(T, range(n - 1 - buff * 2, n - buff))
    prob = np.apply_along_axis(lambda x: np.exp(-(x - tau) ** 2 / sigma2),
                               0, T)
    prob /= np.sum(prob)
    return np.random.choice(T, p=prob)


def kernel_uniform(n, buff):
    """samples the neighbors with the uniform kernel

    Parameters
    ----------

    n : scalar
        sample size
    buff : scalar
        buffer to avoid edges

    Return
    ------
    proposal for changepoint
    """

    return np.random.randint(1 + buff, n - 1 - buff)


def kernel_mixture(tau, n, buff, sigma2, T):
    """samples from the mixture kernel

    Parameters
    ----------

    tau : scalar
        current changepoint
    n : scalar
        sample size
    buff : scalar
        buffer to avoid edges
    sigma2 : scalar
        proposal variance
    T : scalar
        temperature, used to determine how likely it is to draw
        uniform vs gaussian

    Return
    ------
    proposal for changepoint
    """

    if np.random.binomial(1, T / n):
        taup = kernel_uniform(n, buff)
    else:
        taup = kernel_gaussian(tau, n, buff, sigma2)
    return taup


def log_likelihood(data, theta, regularizer):
    """estimates the negative log likelihood for the corresponding
    data subset

    Parameters
    ----------

    data : array-like
        N x P matrix containing the data used for estimation
        should correspond to one side of the change-point
    theta : array-like
        P x P matrix for corresponding data subset
    regularizer : scalar
        regularizing constant

    Returns
    -------
    scalar estimate for the log-likelihood
    """

    n, p = data.shape

    S = np.cov(data.T)
    TdS = theta.dot(S)
    det = np.linalg.slogdet(theta)[1]

    ll = n / 2. * (-det + np.trace(TdS))
    ll += regularizer * np.sqrt(np.log(p) / n) * np.linalg.norm(theta, 1) / 2.

    return -ll
#    return -ll / N


def simulated_annealing(data, theta_init, method, log_likelihood,
                        niter=500, min_beta=1e-4, buff=100,
                        method_kwds=None, ll_kwds=None):
    """estimates a single change-point using the simulated annealing
    algorithm

    Parameters
    ----------

    data : array-like
        N x P matrix containing the data used for estimation
    theta_init : array-like
        starting value for theta estimate
    method : function
        black box method
    log_likelihood : function
        corresponding black-box log-likelihood
    niter : scalar
        number of simulated annealing iterations to run
    min_beta : scalar
        minimum temperature
    buff : scalar
        distance from edge of sample to maintain during search
    method_kwds : None or dict-like
        keywords for method
    ll_kwds : None or dict-like
        keywords for log-likelihood

    Returns
    -------
    tuple containing tau estimate as well as theta estimates
    """

    n, p = data.shape

    tau = np.random.randint(buff * 2, n - buff * 2 - 1)

    if method_kwds is None:
        methods_kwds = {}

    if ll_kwds is None:
        ll_kwds = {}

    taup = tau
    beta = 1
    iterations = 0.

    theta_l = [theta_init, theta_init]

    while beta > min_beta and iterations < niter:

        if tau == taup:
            theta_l[0] = mapping(data[0:tau,:], theta_l[0], **method_kwds)
            theta_l[1] = mapping(data[tau:,:], theta_l[1], **method_kwds)

            ll0 = log_likelihood(data[0:tau,:], theta_l[0], **ll_kwds)
            ll1 = log_likelihood(data[tau:,:], theta_l[1], **ll_kwds)
            ll = ll0 + ll1

        taup = kernel_uniform(n, buff)
        ll0p = log_likelihood(data[0:taup,:], theta_l[0], **ll_kwds)
        ll1p = log_likelihood(data[taup:,:], theta_l[1], **ll_kwds)
        llp = ll0p + ll1p

        prob = np.nanmin(np.array([np.exp((llp - ll) / beta), 1]))

        if prob > np.random.uniform():
            tau = taup

        iterations += 1

        beta = min_beta ** (iterations/niter)

        print iterations, tau, taup, prob, ll, llp, np.linalg.norm(theta_l[0]), np.linalg.norm(theta_l[1]), np.sum(theta_l[0] != 0), np.sum(theta_l[1] != 0), np.sqrt(np.log(p) / data[:taup,:].shape[0]), np.sqrt(np.log(p) / data[taup:,:].shape[0]), beta

    return tau, theta_l


def brute_force(data, buff=10, regularizer=1., update_w=1., mapping_iter=10, epsilon=0.):
    """estimates a single changepoint using the brute force approach

    Parameters
    ----------

    data : array-like
        N x P matrix containing the data used for estimation
    buff : scalar
        buffer to keep proposals away from boundary
    regularizer : scalar
        regularizing constant
    mapping_kwds : None or dict-like
        key words for mapping fn

    Returns
    -------
    tuple of tau_l (list) theta_l (list of arrays) and ll (scalar)
    """

    n, p = data.shape

    mapping_kwds = {"update_w": update_w,
                    "update_change": 0.9,
                    "regularizer": regularizer,
                    "mapping_iter": mapping_iter,
                    "tol": 0.0000001}

    S_inv = np.linalg.inv(np.cov(data.T) + np.eye(p) * epsilon)

    theta_l = [S_inv, S_inv]

    tau_l = []
    ll_l = []

    for taup in range(buff, n - buff):

        theta_l = [S_inv, S_inv]
        S0 = np.cov(data[:taup,:].T)
        mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[:taup,:].shape[0])
#        mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[:taup,:].shape[0])
        theta0 = mapping(S0, theta_l[0], **mapping_kwds)
        S1 = np.cov(data[taup:,:].T)
        mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[taup:,:].shape[0])
#        mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[taup:,:].shape[0])
        theta1 = mapping(S1, theta_l[1], **mapping_kwds)
        theta_l = [theta0, theta1]

        ll = log_likelihood(data, theta_l, [0, taup, n], regularizer, n)
        ll_l.append(ll)
        tau_l.append(taup)
        print taup, np.linalg.norm(theta0), np.linalg.norm(theta1), np.sum(theta0 != 0), np.sum(theta1 != 0), ll

    ind = np.argmax(ll_l)
    tau = tau_l[ind]

    theta_l = [S_inv, S_inv]
    S0 = np.cov(data[:tau,:].T)
    mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[:tau,:].shape[0])
#    mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[:tau,:].shape[0])
    theta0 = mapping(S0, theta_l[0], **mapping_kwds)
    S1 = np.cov(data[tau:,:].T)
    mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[tau:,:].shape[0])
#    mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[tau:,:].shape[0])
    theta1 = mapping(S1, theta_l[1], **mapping_kwds)
    theta_l = [theta0, theta1]

    ll = ll_l[ind]

#    plt.plot(ll)
#    plt.savefig("test.pdf")
#    plt.clf()

    return tau_l + [tau], theta_l, ll, ll_l


def log_likelihood_rank_one(data, S_l, theta_l, buff, tau, regularizer, iteration):
    """handles the rank one log likelihood estimation and selection
    of new changepoints based on the log likelihood

    Parameters
    ----------

    data : array-like
        N x P matrix of data used for estimation
    S_l : list of array-likes
        length 2 list containing inverse covariance estimates
    theta_l : list of array-likes
        length 2 list containing estimates for theta
    buff : scalar
        buffer to keep from edges
    tau : scalar
        current changepoint estimate
    regularizer : scalar
        regularizing constant

    Returns
    -------
    tuple containing tau and ll
    """

    n, p = data.shape

    TdS_l = [theta_i.dot(S_i) for theta_i, S_i in zip(theta_l, S_l)]

    det_l = [np.linalg.slogdet(theta_i)[1] for theta_i in theta_l]

    ll = (tau / (2.) * (-det_l[0] + np.trace(TdS_l[0])) +
                        np.sqrt(np.log(p) / tau) * regularizer * np.linalg.norm(theta_l[0], 1))
    ll += ((n - tau) / (2.) * (-det_l[1] + np.trace(TdS_l[1])) +
                               np.sqrt(np.log(p) / (n - tau)) * regularizer * np.linalg.norm(theta_l[1], 1))
#    ll = ((-det_l[0] + np.trace(TdS_l[0])) +
#          np.sqrt(np.log(p) / tau) * regularizer * np.linalg.norm(theta_l[0], 1))
#    ll += ((-det_l[1] + np.trace(TdS_l[1])) +
#           np.sqrt(np.log(p) / (n - tau)) * regularizer * np.linalg.norm(theta_l[1], 1))
    ll_l = [-ll]

    Sp_l = [S_i[:] for S_i in S_l]
    TdSp_l = [theta_i.dot(Sp_i) for theta_i, Sp_i in zip(theta_l, Sp_l)]
    ll_l1 = []

    for i in range(tau - 1, buff, -1):
#        Sp_l = [data[:i,:].T.dot(data[:i,:]) / data[:i,:].shape[0],
#                data[i:,:].T.dot(data[i:,:]) / data[i:,:].shape[0]]
#        Sp_l = [np.cov(data[:i,:].T), np.cov(data[i:,:].T)]
#        mn1 = np.mean(data[:i,:], axis=0)
#        mn2 = np.mean(data[i:,:], axis=0)
#        roneupdate1 = np.outer((data[i,:] - mn1), (data[i,:] - mn1))
#        roneupdate2 = np.outer((data[i,:] - mn2), (data[i,:] - mn2))
        rank_one_update = np.outer(data[i,:], data[i,:])

        Sp_l = [((i) * Sp_l[0] - rank_one_update) / (i - 1),
                ((n - i - 1) * Sp_l[1] + rank_one_update) / (n - i)]
#        Sp_l = [Sp_l[0] - rank_one_update, Sp_l[1] + rank_one_update]
#        Sp_l = [np.cov(data[:i,:].T), np.cov(data[i:,:].T)]


        TdSp_l = [theta_i.dot(Sp_i) for theta_i, Sp_i in zip(theta_l, Sp_l)]
#        TdSp_l = [TdSp_l[0] - theta_l[0].dot(rank_one_update),
#                  TdSp_l[1] + theta_l[1].dot(rank_one_update)]

        ll = (i / (2.) * (-det_l[0] + np.trace(TdSp_l[0])) +
                          np.sqrt(np.log(p) / i) * regularizer * np.linalg.norm(theta_l[0], 1))
        ll += ((n - i) / (2.) * (-det_l[1] + np.trace(TdSp_l[1])) +
                                 np.sqrt(np.log(p) / (n - i)) * regularizer * np.linalg.norm(theta_l[1], 1))
#        ll = ((-det_l[0] + np.trace(TdSp_l[0])) +
#              np.sqrt(np.log(p) / i) * regularizer * np.linalg.norm(theta_l[0], 1))
#        ll += ((-det_l[1] + np.trace(TdSp_l[1])) +
#               np.sqrt(np.log(p) / (n - i)) * regularizer * np.linalg.norm(theta_l[1], 1))
        ll_l1.append(-ll)

    ll_l1.reverse()
    Sp_l = [S_i[:] for S_i in S_l]
    TdSp_l = [theta_i.dot(Sp_i) for theta_i, Sp_i in zip(theta_l, Sp_l)]
    ll_l2 = []

    for i in range(tau + 1, n - buff):
#        Sp_l = [data[:i,:].T.dot(data[:i,:]) / data[:i,:].shape[0],
#                data[i:,:].T.dot(data[i:,:]) / data[i:,:].shape[0]]
#        Sp_l = [np.cov(data[:i,:].T), np.cov(data[i:,:].T)]
        rank_one_update = np.outer(data[i,:], data[i,:])

        Sp_l = [((i - 1) * Sp_l[0] + rank_one_update) / (i),
                ((n - i) * Sp_l[1] - rank_one_update) / (n - i - 1)]
#        Sp_l = [Sp_l[0] + rank_one_update, Sp_l[1] - rank_one_update]
#        Sp_l = [np.cov(data[:i,:].T), np.cov(data[i:,:].T)]

        TdSp_l = [theta_i.dot(Sp_i) for theta_i, Sp_i in zip(theta_l, Sp_l)]
#        TdSp_l = [TdSp_l[0] + theta_l[0].dot(rank_one_update),
#                  TdSp_l[1] - theta_l[1].dot(rank_one_update)]

        ll = (i / (2.) * (-det_l[0] + np.trace(TdSp_l[0])) +
                          np.sqrt(np.log(p) / i) * regularizer * np.linalg.norm(theta_l[0], 1))
        ll += ((n - i) / (2.) * (-det_l[1] + np.trace(TdSp_l[1])) +
                                 np.sqrt(np.log(p) / (n - i)) * regularizer * np.linalg.norm(theta_l[1], 1))
#        ll = ((-det_l[0] + np.trace(TdSp_l[0])) +
#              np.sqrt(np.log(p) / i) * regularizer * np.linalg.norm(theta_l[0], 1))
#        ll += ((-det_l[1] + np.trace(TdSp_l[1])) +
#               np.sqrt(np.log(p) / (n - i)) * regularizer * np.linalg.norm(theta_l[1], 1))
        ll_l2.append(-ll)

    ll_l = ll_l1 + ll_l + ll_l2

#    print np.argmax(ll_l)
#    print ll_l[20:130]

    ind = np.argmax(ll_l)

    plt.plot(ll_l)
    plt.savefig("ro_%d_%d.pdf" % (ind + buff, iteration))
    plt.clf()

    return ind + buff + 1, ll_l[ind]


def rank_one(data, buff=10, regularizer=1., tau=-1, max_iter=25,
             update_w=1, mapping_iter=1):
    """estimates the single changepoint using the rank-one method

    data : array-like
        N x P matrix containing the data used for estimation
    buff : scalar
        buffer to keep proposals away from boundary
    regularizer : scalar
        regularizing constant
    tau : scalar
        starting proposal for tau, if -1 is selected uniformly
    max_iter : scalar
        the maximum number of iterations to allow, note that this is given
        as a float (0, 1) and is multiplied by n to get the actual number
    mapping_kwds : None or dict-like
        key words for mapping fn

    Returns
    -------
    tuple of tau (scalar) theta_l (list of arrays) and ll (scalar)
    """

    n, p = data.shape

    if tau == -1:
        tau = np.random.randint(buff, n - buff - 1)

    mapping_kwds = {"update_w": update_w,
                    "update_change": 0.9,
                    "regularizer": regularizer,
                    "mapping_iter": mapping_iter,
                    "tol": 0.000001}

    iterations = 0
    taup = -1

    S_inv = np.linalg.inv(np.cov(data.T) + 1)

    theta_l = [S_inv, S_inv]

    tau_l = []

    while iterations < max_iter:

#        theta_l = [S_inv, S_inv]
        S0 = np.cov(data[:tau,:].T)
        mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[:tau,:].shape[0])
#        mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[:tau,:].shape[0])
        theta0 = mapping(S0, theta_l[0], **mapping_kwds)
        S1 = np.cov(data[tau:,:].T)
        mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[tau:,:].shape[0])
#        mapping_kwds["update_w"] = update_w * np.sqrt(np.log(p) / data[tau:,:].shape[0])
        theta1 = mapping(S1, theta_l[1], **mapping_kwds)
        theta_l = [theta0, theta1]

#        S_l = [data[:tau,:].T.dot(data[:tau,:]) / data[:tau,:].shape[0],
#               data[tau:,:].T.dot(data[tau:,:]) / data[tau:,:].shape[0]]
        S_l = [np.cov(data[:tau,:].T), np.cov(data[tau:,:].T)]

        tau, ll = log_likelihood_rank_one(data, S_l, theta_l, buff, tau,
                                          regularizer, iterations)
        log_likelihood(data, theta_l, [0, tau, n], regularizer, n)

#        ll_l = []
#        for stp in range(buff, n - buff):
#            ll_l.append(log_likelihood(data, theta_l, [0, stp, n], regularizer, n))
#        plt.plot([l + buff for l in ll_l])
#        plt.savefig("%d_%d.pdf" % (iterations, tau))
#        plt.clf()
        iterations += 1
        tau_l.append(tau)
        print iterations, ll, np.sum(theta_l[0] == 0), np.sum(theta_l[1] == 0), np.linalg.norm(theta_l[0]), np.linalg.norm(theta_l[1]), tau

    return tau_l, theta_l, ll


def cp_data_ll(data, tau, regularizer=1., update_w=1., mapping_iter=1.):
    """returns the likelihood for the change point data"""

    n, p = data.shape
    mapping_kwds = {"update_w": update_w,
                    "update_change": 0.9,
                    "regularizer": regularizer,
                    "mapping_iter": mapping_iter,
                    "tol": 0.000001}

    S = np.cov(data.T)
    S_inv = np.linalg.inv(S)
    theta_l = [S_inv, S_inv]
    S0 = np.cov(data[:tau,:].T)
    mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[:tau,:].shape[0])
    theta0 = mapping(S0, theta_l[0], **mapping_kwds)
    S1 = np.cov(data[tau:,:].T)
    mapping_kwds["regularizer"] = regularizer * np.sqrt(np.log(p) / data[tau:,:].shape[0])
    theta1 = mapping(S1, theta_l[1], **mapping_kwds)
    theta_l = [theta0, theta1]

    print np.linalg.norm(theta0), np.linalg.norm(theta1), np.sum(theta0 != 0), np.sum(theta1 != 0)

    ll = log_likelihood(data, theta_l, [0, tau, n], regularizer, n)
    return ll


def full_data_ll(data, regularizer=None, mapping_iter=None, mapping_kwds=None):
    """returns the likelihood for the full data"""

    n, p = data.shape

    if regularizer is None:
        regularizer = np.sqrt(np.log(p) / n)

    if mapping_iter is None:
        mapping_iter = 1
    update_w = regularizer

    if mapping_kwds is None:
        mapping_kwds = {"update_w": update_w,
                        "update_change": 0.9,
                        "regularizer": regularizer,
                        "mapping_iter": mapping_iter,
                        "tol": 0.000001}

    n, p = data.shape
    S = np.cov(data.T)
    theta = mapping(S, np.eye(p), **mapping_kwds)
    det = np.linalg.slogdet(theta)[1]
    TdS = theta.dot(S)
    ll = (det - np.trace(TdS) + regularizer * np.linalg.norm(theta, 1))
#    ll = det - np.trace(TdS)
    return ll


def binary_segmentation(data, thresh, method, buff, method_kwds=None,
                        mapping_kwds=None):
    """ performs the binary segmentation using simulated annealing
    Parameters
    ----------
    data : array-like
        matrix of covariates
    thresh : scalar
        the threshold used for binary segmentation
    method : function
        method for estimating changepoints
    method_kwds : dict-like or None
        key word arguments for method
    mapping_kwds : dict-like or None
        the keywords for the mapping function
    Returns
    -------
    tuple of tau_l, theta_l, ll
    """

    n, p = data.shape

    edge_check = float(buff) / n

    buff = 25

    if method_kwds is None:
        method_kwds = {}

    # a list of all the changepoints found
    cp = [0, n]
    # records which regions we belive still need a changepoint, one
    # indicates that a changepoint could be added
    state = [1]

    while sum(state) > 0:
        # first we build the log likelihood that will be compared
        cpt = [0]
        statet = []
        for i, c in enumerate(cp[1:]):
            # the way that this loop works, cp[i] and c always end up
            # being off by one so it should be able to handle the
            # area between changepoints
            if state[i]:
                datat = data[cp[i]:c,:]
                nt = datat.shape[0]
                if nt > 2 * (buff + 1):
#                    tautl, thetalt, llt = method(datat, buff=int(edge_check * nt), mapping_kwds=mapping_kwds, **method_kwds)
                    tautl, thetalt, llt = simulated_annealing(datat, buff=10, mapping_kwds=mapping_kwds, regularizer=0.1, update_w=0.1)
                    taut = tautl[-1]
#                    mapping_iter = len(set(tautl))
                    mapping_iter = 25
                    llf = full_data_ll(datat, mapping_iter=mapping_iter, mapping_kwds=mapping_kwds)
                else:
                    llf = float("inf")
                    llt = 0
                cond1 = (llt - llf) / nt < thresh * p / np.log(len(cp) - 1)
#                print llt, llf, (llt - llf) / nt, thresh * p / np.log(len(cp) - 1)
                print llt, llf, thresh * p / np.log(len(cp) - 1), (llt - llf) / nt, cond1, taut, nt - nt * edge_check, nt * edge_check
#                if cond1 and taut < (1 - edge_check) * nt  and taut > nt * edge_check:
                if cond1 and taut < nt - buff and taut > buff:
                    cpt.append(taut + cp[i])
                    cpt.append(c)
                    statet.append(1)
                    statet.append(1)
                else:
                    cpt.append(c)
                    statet.append(0)
            else:
                cpt.append(c)
                statet.append(0)

        cp = cpt
        state = statet

    cp = sorted(cp[1:-1])
    return cp
