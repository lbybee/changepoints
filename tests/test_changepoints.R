library(changepoints)

set.seed(42)


t_data <- rnorm(1000)
t_data <- matrix(t_data, nrow=100, ncol=10)

prox_gradient_params=list()
prox_gradient_params$update_w = 0.1
prox_gradient_params$update_change = 0.99
prox_gradient_params$regularizer = 0.1
prox_gradient_params$max_iter = 2
prox_gradient_params$tol = 1e-5

prox_gradient_ll_params=list()
prox_gradient_ll_params$regularizer = 0.1

simulated_annealing_params=list()
simulated_annealing_params$buff=10

res_sa = simulated_annealing(t_data, diag(10), prox_gradient_mapping,
                             prox_gradient_ll, buff=10,
                             bbmod_method_params=prox_gradient_params,
                             bbmod_ll_params=prox_gradient_ll_params)
res_bf = brute_force(t_data, diag(10), prox_gradient_mapping,
                     prox_gradient_ll, buff=10,
                     bbmod_method_params=prox_gradient_params,
                     bbmod_ll_params=prox_gradient_ll_params)
res_ro = rank_one(t_data, diag(10))
res_bs = binary_segmentation(t_data, diag(10), simulated_annealing,
                             prox_gradient_mapping,
                             prox_gradient_ll,
                             cp_method_params=simulated_annealing_params,
                             bbmod_method_params=prox_gradient_params,
                             bbmod_ll_params=prox_gradient_ll_params)

#mod = ChangepointModel(data, bbmod_params, bbmod_ll_params,
#                       mapping_wrapper, ll_wrapper)

#sa_res = mod.simulated_annealing()
