from datetime import datetime
from changepoints import *
import numpy as np

t1 = datetime.now()

np.random.seed(334)

scp_data = np.loadtxt("scp.txt")

# prox gradient black-box method
cov_est = np.cov(scp_data.T)
init = np.linalg.inv(cov_est)
res_map = mapping(scp_data, init, 0.1, 0.99, 0.1, 100, 1e-4)

# prox gradient black-box ll
res_ll = log_likelihood(scp_data, res_map, 0.1)

# simulated annealing with prox gradient
method_kwds = {"update_w": 0.1, "update_change": 0.99,
        "regularizer": 0.1, "mapping_iter": 1,
        "tol": 1e-5}
ll_kwds = {"regularizer": 0.1}
res_sa = simulated_annealing(scp_data, init, mapping, log_likelihood, buff=10,
                             method_kwds=method_kwds, ll_kwds=ll_kwds)

# brute force with prox gradient
res_bf = brute_force(scp_data, init, mapping, log_likelihood, buff=10,
                     method_kwds=method_kwds, ll_kwds=ll_kwds)

# rank one with prox gradient
res_ro = rank_one(scp_data, init, regularizer=0.1, update_w=0.1, buff=10,
                  max_iter=5)


t2 = datetime.now()
print t2 - t1
#res = simulated_annealing(scp_data, Tmin=1./100, Tmax=1., regularizer=0.1,
#                          update_w=0.1, buff=10, max_iter=500)

#res = rank_one(scp_data, regularizer=0.1, update_w=0.1, buff=10, max_iter=5)

#res = brute_force(scp_data, buff=10, mapping_iter=5)

#mcp_data = np.loadtxt("mcp.txt")

#method_kwds = {"regularizer": 0.1, "update_w": 0.1}
#res = binary_segmentation(mcp_data, 0, simulated_annealing, 10,
#                          method_kwds=method_kwds)
