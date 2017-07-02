library(changepoints)

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

res_sa = simulated_annealing(scp_data, init, prox_gradient_mapping,
                             prox_gradient_ll, buff=10,
                             bbmod_method_params=prox_gradient_params,
                             bbmod_ll_params=prox_gradient_ll_params)
res_bf = brute_force(scp_data, init, prox_gradient_mapping,
                     prox_gradient_ll, buff=10,
                     bbmod_method_params=prox_gradient_params,
                     bbmod_ll_params=prox_gradient_ll_params)
#res_ro = rank_one(t_data, diag(10))
#res_bs = binary_segmentation(t_data, diag(10), simulated_annealing,
#                             prox_gradient_mapping,
#                             prox_gradient_ll,
#                             cp_method_params=simulated_annealing_params,
#                             bbmod_method_params=prox_gradient_params,
#                             bbmod_ll_params=prox_gradient_ll_params)

# comparable tests to the above for LDA
t_data <- sample(0:100, 1000, replace=TRUE)
t_data <- matrix(t_data, nrow=100, ncol=10)

corpus = t_data
latent_vars=list()
z=matrix(sample(0:1, 1000, replace=TRUE), nrow=100, ncol=10)
latent_vars$theta=matrix(0, nrow=100, ncol=2)
latent_vars$phi=matrix(0, nrow=2, ncol=10)
nw=matrix(0, nrow=10, ncol=2)
nd=matrix(0, nrow=100, ncol=2)
nwsum=numeric(2)
ndsum=numeric(100)
for(d in 1:100){
    for(v in 1:10){
        topic = z[d,v]
        vcount = corpus[d,v]
        nw[v,topic] = nw[v,topic] + vcount
        nd[d,topic] = nd[d,topic] + vcount
        nwsum[topic] = nwsum[topic] + vcount
        ndsum[d] = ndsum[d] + vcount
    }
}
latent_vars$z = z
latent_vars$nw = nw
latent_vars$nd = nd
latent_vars$nwsum = nwsum
latent_vars$ndsum = ndsum

latent_dirichlet_allocation_params=list()
latent_dirichlet_allocation_params$niter = 50
latent_dirichlet_allocation_params$alpha = 1
latent_dirichlet_allocation_params$beta = 1

#res_sa = simulated_annealing(corpus, latent_vars, latent_dirichlet_allocation,
#                             latent_dirichlet_allocation_ll, buff=10,
#                             bbmod_method_params=latent_dirichlet_allocation_params)
#res_bf = simulated_annealing(corpus, latent_vars, latent_dirichlet_allocation,
#                             latent_dirichlet_allocation_ll, buff=10,
#                             bbmod_method_params=latent_dirichlet_allocation_params)
#res_bs = binary_segmentation(corpus, latent_vars, simulated_annealing,
#                             latent_dirichlet_allocation,
#                             latent_dirichlet_allocation_ll,
#                             cp_method_params=simulated_annealing_params,
#                             bbmod_method_params=latent_dirichlet_allocation_params)

# comparable tests to the above showing the use of this approach
# for not included models (GLM).
