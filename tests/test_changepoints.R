library(changepoints)

set.seed(42)

mapping_wrapper <- function(data, bbmod_params){

    cov_est <- cov(data)
    theta_start <- bbmod_params$theta_start
    update_w <- bbmod_params$update_w
    update_change <- bbmod_params$update_change
    regularizer <- bbmod_params$regularizer
    max_iter <- bbmod_params$max_iter
    tol <- bbmod_params$tol

    theta_est <- mapping(cov_est, theta_start, update_w, update_change,
                         regularizer, max_iter, tol)

    return(theta_est)
#    res <- list(theta_est)
#    names(res) <- "theta_est"
#    return(res)
}


ll_wrapper <- function(data, bbmod_ll_params){

    theta_i <- bbmod_ll_params$theta_i
    regularizer <- bbmod_ll_params$regularizer

    ll = gg_log_likelihood(data, theta_i, regularizer)

    return(ll)
}


mod = ChangepointModel(data, bbmod_params, bbmod_ll_params,
                       mapping_wrapper, ll_wrapper)

sa_res = mod.simulated_annealing()
