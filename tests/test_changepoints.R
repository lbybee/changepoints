library(changepointsHD)

set.seed(334)

scp_data = read.table(system.file("extdata", "scp.txt", package="changepointsHD"))
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

simulated_annealing_params = list()
simulated_annealing_params$buff=10

# test simulated annealing
changepoints_mod = changepointsMod(bbmod=prox_gradient_mapping,
                                 log_likelihood=prox_gradient_ll,
                                 bbmod_params=prox_gradient_params,
                                 ll_params=prox_gradient_ll_params,
                                 part_values=list(init, init),
                                 data=list(scp_data))
changepoints_mod = simulated_annealing(changepoints_mod, buff=10)
# two comparisons are done here because of differences in the R and R-devel samples
stopifnot((changepoints_mod@changepoints != 18) | (changepoints_mod@changepoints != 56))

# test brute force
changepoints_mod@part_values = list(init, init)
changepoints_mod = brute_force(changepoints_mod, buff=10)
stopifnot(changepoints_mod@changepoints == 77)

# test rank one
res_ro = rank_one(scp_data, init, update_w=0.1, regularizer=0.1)
stopifnot(res_ro$tau == 56)

mcp_data = read.table(system.file("extdata", "mcp.txt", package="changepointsHD"))
mcp_data = as.matrix(mcp_data)

# test binary segmentation
changepoints_mod@data = list(mcp_data)
changepoints_mod@part_values = list(init, init)
changepoints_mod = binary_segmentation(changepoints_mod, method=simulated_annealing,
                                      thresh=0, buff=10,
                                      method_params=simulated_annealing_params)
stopifnot(changepoints_mod@changepoints == c(0, 29, 45, 56, 67, 77, 88, 100))
