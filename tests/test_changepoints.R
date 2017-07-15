library(changepointsHD)

set.seed(334)

scp_data = read.table("scp.txt")
scp_data = as.matrix(scp_data)

# prox gradient black-box method
cov_est = cov(scp_data)
init = solve(cov_est)
res_map = prox_gradient_mapping(scp_data, init, 0.1, 0.99, 0.1, 100, 1e-20)

# prox gradient black-box ll
res_ll = prox_gradient_ll(scp_data, res_map, 0.1)

prox_gradient_params=list()
prox_gradient_params$update_w = 0.1
prox_gradient_params$update_change = 0.99
prox_gradient_params$regularizer = 0.1
prox_gradient_params$max_iter = 1
prox_gradient_params$tol = 1e-5

prox_gradient_ll_params=list()
prox_gradient_ll_params$regularizer = 0.1

simulated_annealing_params=list()
simulated_annealing_params$buff=10

# test simulated annealing
res_sa = simulated_annealing(scp_data, init, prox_gradient_mapping,
                             prox_gradient_ll, buff=10,
                             bbmod_method_params=prox_gradient_params,
                             bbmod_ll_params=prox_gradient_ll_params)
stopifnot(res_sa$tau == 56)

# test brute force
res_bf = brute_force(scp_data, init, prox_gradient_mapping,
                     prox_gradient_ll, buff=10,
                     bbmod_method_params=prox_gradient_params,
                     bbmod_ll_params=prox_gradient_ll_params)
stopifnot(res_bf$tau == 43)

# test rank one
res_ro = rank_one(scp_data, init, update_w=0.1, regularizer=0.1)
stopifnot(res_ro$tau == 56)

mcp_data = read.table("mcp.txt")
mcp_data = as.matrix(mcp_data)

# test binary segmentation
res_bs = binary_segmentation(mcp_data, init, simulated_annealing,
                             prox_gradient_mapping,
                             prox_gradient_ll, buff=10,
                             cp_method_params=simulated_annealing_params,
                             bbmod_method_params=prox_gradient_params,
                             bbmod_ll_params=prox_gradient_ll_params)
stopifnot(res_bs$tau_l == c(0, 29, 45, 56, 67, 77, 88, 100))
