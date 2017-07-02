from changepoints import *
import scipy.sparse as ss
import numpy as np

np.random.seed(334)

def gen_data(n, p, cp_ln):

    cov_l = []
    prec_l = []

    for j in range(cp_ln):
        a = ss.rand(p, p, density=0.125).A
        m = a + a.T
        m[m < 0] -= 4
        m[m > 0] += 4
        mineig = np.min(np.linalg.eigvals(m))
        np.fill_diagonal(m, (1 + np.diag(m) - mineig))
        prec_l.append(m)
        cov_l.append(np.linalg.inv(m))

    data_l = []
    for cov in cov_l:
        data_l.append(np.random.multivariate_normal(np.zeros(p), cov,
                                                    n / cp_ln))
    data = np.concatenate(data_l)

    return data


n = 100
p = 10
cp_ln = 2

data = gen_data(n, p, cp_ln)

np.savetxt("scp.txt", data)

cp_ln = 4

data = gen_data(n, p, cp_ln)

np.savetxt("mcp.txt", data)
