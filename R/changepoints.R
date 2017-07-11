#' @name simulated_annealing
#'
#' @title Single change-point simulated annealing method
#'
#' @description Estimates a single change-point using the simulated annealing
#'              method.
#'
#' @param data Matrix of actual data.  Should be N x P where N is the number
#'             of observations and p is the number of variables.  Note that
#'             given the way this is passed to \code{bbmod_method} and \code{bbmod_ll}
#'             this could possibly handle more complex data structures.
#'             The key is that we assume that there are N rows.
#'             This is key because me must subset data before passing
#'             to \code{bbmod_method} and \code{bbmod_ll}.
#' @param bbmod_init_vals Initial estimates for black box model values.
#'                        In the GGM case these correspond to a P x P precision
#'                        matrix.  However, as with \code{data} it is possible to
#'                        handle more complex data structures.  For instance,
#'                        in the topic modeling case this is a list containing
#'                        the current values for the many latent variables
#'                        used in latent Dirichlet allocation.
#' @param bbmod_method Function corresponding to black box model itself.
#'                     Should return an updated version of \code{bbmod_vals}.
#' @param bbmod_ll Function for ll or comparable cost function for
#'                 black box model.  Should return scalar estimate of ll/cost.
#' @param niter Number of simulated annealing iterations.
#' @param min_beta Lowest temperature.
#' @param buff Distance from edge of sample to be maintained during search.
#' @param bbmod_method_params List of additional parameters for \code{bbmod_method}.
#' @param bbmod_ll_params List of additional parameters for \code{bbmod_ll}.
#'
#' @return List containing estimated change-point and black box model estimates.
#'
#' @author \packageMaintainer{changepointsHD}
simulated_annealing <- function(data, bbmod_init_vals, bbmod_method, bbmod_ll,
                                niter=500, min_beta=1e-4, buff=100,
                                bbmod_method_params=list(),
                                bbmod_ll_params=list()){
    "simulated annealing for esitmating change-points, returns a list
    containing the location of the estimated change-point as well as the
    estimated black box model values
    "

    # TODO add support for different init vals
    bbmod_vals = list(bbmod_init_vals, bbmod_init_vals)

    # initialize parameters
    N = dim(data)[1]
    ptaus = buff:(N-buff)
    tau = sample(ptaus, 1)
    taup = tau
    iterations = 0
    beta = 1
    change = TRUE

    while((beta > min_beta) & (iterations < niter)){

        if(change){

            bbmod_vals[[1]] = do.call(bbmod_method, c(list(data[1:tau,],
                                                           bbmod_vals[[1]]),
                                                      bbmod_method_params))
            bbmod_vals[[2]] = do.call(bbmod_method, c(list(data[(tau+1):N,],
                                                           bbmod_vals[[2]]),
                                                      bbmod_method_params))

            ll0 = do.call(bbmod_ll, c(list(data[1:tau,], bbmod_vals[[1]]),
                                      bbmod_ll_params))
            ll1 = do.call(bbmod_ll, c(list(data[(tau+1):N,], bbmod_vals[[2]]),
                                      bbmod_ll_params))
            ll = ll0 + ll1
            change = FALSE
        }

        taup = sample(ptaus[-which(ptaus == tau)], 1)
        ll0p = do.call(bbmod_ll, c(list(data[1:taup,], bbmod_vals[[1]]),
                                   bbmod_ll_params))
        ll1p = do.call(bbmod_ll, c(list(data[(taup+1):N,], bbmod_vals[[2]]),
                                   bbmod_ll_params))
        llp = ll0p + ll1p

        prob = min(1, exp((llp - ll) / beta))

        u = runif(1)

        print(paste("Iteration:", iterations, "LL:", ll, "Prop LL:", llp, "CP:", tau, "Prop CP:", taup))
        if (prob > u){
            tau = taup
            change = TRUE
        }

        beta = min_beta^(iterations/niter)
        iterations = iterations + 1
    }
    res = list()
    res$tau = tau
    res$bbmod_vals = bbmod_vals
    return(res)
}


#' @name brute_force
#'
#' @title Single change-point brute force method.
#'
#' @description Estimates a single change-point by testing all possible
#'              change-points.
#'
#' @param data Matrix of actual data.  Should be N x P where N is the number
#'             of observations and p is the number of variables.  Note that
#'             given the way this is passed to \code{bbmod_method} and \code{bbmod_ll}
#'             this could possibly handle more complex data structures.
#'             The key is that we assume that there are N rows.
#'             This is key because me must subset data before passing
#'             to \code{bbmod_method} and \code{bbmod_ll}.
#' @param bbmod_init_vals Initial estimates for black box model values.
#'                        In the GGM case these correspond to a P x P precision
#'                        matrix.  However, as with \code{data} it is possible to
#'                        handle more complex data structures.  For instance,
#'                        in the topic modeling case this is a list containing
#'                        the current values for the many latent variables
#'                        used in latent Dirichlet allocation.
#' @param bbmod_method Function corresponding to black box model itself.
#'                     Should return an updated version of \code{bbmod_vals}.
#' @param bbmod_ll Function for ll or comparable cost function for
#'                 black box model.  Should return scalar estimate of ll/cost.
#' @param buff Distance from edge of sample to be maintained during search.
#' @param bbmod_method_params List of additional parameters for \code{bbmod_method}.
#' @param bbmod_ll_params List of additional parameters for \code{bbmod_ll}.
#'
#' @return List containing estimated change-point and black box model estimates.
#'
#' @author \packageMaintainer{changepointsHD}

brute_force <- function(data, bbmod_init_vals, bbmod_method, bbmod_ll,
                        buff=100, bbmod_method_params=list(),
                        bbmod_ll_params=list()){
    "brute force method for estimating change-points, returns the index of
    the estimated change-point
    "

    bbmod_vals = list(bbmod_init_vals, bbmod_init_vals)

    N = dim(data)[1]
    ll_l = c()

    for(i in buff:(N-buff)){
        bbmod_vals0 = do.call(bbmod_method, c(list(data[1:i,],
                                                   bbmod_vals[[1]]),
                                              bbmod_method_params))
        bbmod_vals1 = do.call(bbmod_method, c(list(data[(i+1):N,],
                                                   bbmod_vals[[2]]),
                                              bbmod_method_params))
        ll0 = do.call(bbmod_ll, c(list(data[1:i,], bbmod_vals[[1]]),
                                  bbmod_ll_params))
        ll1 = do.call(bbmod_ll, c(list(data[(i+1):N,], bbmod_vals[[1]]),
                                  bbmod_ll_params))
        ll = ll0 + ll1
        ll_l = c(ll_l, ll)
        print(paste("LL:", ll, "CP:", i))
    }

    tau = which.min(ll_l)
    bbmod_vals0 = do.call(bbmod_method, c(list(data[1:i,],
                                               bbmod_vals[[1]]),
                                          bbmod_method_params))
    bbmod_vals1 = do.call(bbmod_method, c(list(data[(i+1):N,],
                                               bbmod_vals[[2]]),
                                          bbmod_method_params))
    bbmod_vals = list(bbmod_vals0, bbmod_vals1)
    res = list()
    res$tau = tau
    res$bbmod_vals = bbmod_vals
    return(res)
}


#' @name binary_segmentation
#'
#' @title Multiple change-point method.
#'
#' @description Estimates multiple change-points using the binary-segmentation
#'              method.  This does a breadth first search and uses the specified
#'              single change-point method for each sub-search.
#'
#' @param data Matrix of actual data.  Should be N x P where N is the number
#'             of observations and p is the number of variables.  Note that
#'             given the way this is passed to \code{bbmod_method} and \code{bbmod_ll}
#'             this could possibly handle more complex data structures.
#'             The key is that we assume that there are N rows.
#'             This is key because me must subset data before passing
#'             to \code{bbmod_method} and \code{bbmod_ll}.
#' @param bbmod_init_vals Initial estimates for black box model values.
#'                        In the GGM case these correspond to a P x P precision
#'                        matrix.  However, as with \code{data} it is possible to
#'                        handle more complex data structures.  For instance,
#'                        in the topic modeling case this is a list containing
#'                        the current values for the many latent variables
#'                        used in latent Dirichlet allocation.
#' @param cp_method Function for finding single change-point.
#' @param bbmod_method Function corresponding to black box model itself.
#'                     Should return an updated version of \code{bbmod_vals}.
#' @param bbmod_ll Function for ll or comparable cost function for
#'                 black box model.  Should return scalar estimate of ll/cost.
#' @param thresh Stopping threshold for cost comparison.
#' @param buff Distance from edge of sample to be maintained during search.
#' @param cp_method_params List of additional parameters for \code{cp_method}.
#' @param bbmod_method_params List of additional parameters for \code{bbmod_method}.
#' @param bbmod_ll_params List of additional parameters for \code{bbmod_ll}.
#'
#' @return List containing estimated change-points and model estimates.
#'
#' @author \packageMaintainer{changepointsHD}
binary_segmentation <- function(data, bbmod_init_vals, cp_method,
                                bbmod_method, bbmod_ll, thresh=0, buff=100,
                                cp_method_params=list(),
                                bbmod_method_params=list(),
                                bbmod_ll_params=list()){
    "handles the binary segmentation
    "

    bbmod_vals_base = list(bbmod_init_vals)

    d = dim(data)
    N = d[1]
    P = d[2]

    cp_l = c(0, N)
    ll_l = c(-1e20)
    state_l = c(1)

    while(sum(state_l) > 0){

        t_cp_l = c(0)
        t_ll_l = c()
        t_state_l = c()
        t_bbmod_vals_base = list()

        for(i in 1:length(state_l)){

            if(state_l[i] == 1){

                datat = data[cp_l[i]:cp_l[i+1],]
                Nt = dim(datat)[1]

                if(Nt > 2 * (buff + 1)){

                    cp_method_params$bbmod_method_params = bbmod_method_params
                    cp_method_params$bbmod_ll_params = bbmod_ll_params

                    tres = do.call(cp_method, c(list(datat, bbmod_init_vals,
                                                   bbmod_method, bbmod_ll),
                                                   cp_method_params))
                    tau = tres$tau
                    bbmod_vals = tres$bbmod_vals

                    ll0 = do.call(bbmod_ll, c(list(datat[1:tau,],
                                                   bbmod_vals[[1]]),
                                              bbmod_ll_params))
                    ll1 = do.call(bbmod_ll, c(list(datat[(tau+1):Nt,],
                                                   bbmod_vals[[2]]),
                                              bbmod_ll_params))
                    ll = ll0 + ll1

                    cond1 = (ll - ll_l[i]) > thresh * P
                    # TODO, this doesn't directly follow the approach in the
                    # paper, the issue is that we don't want to make
                    # assumptions about the black box model structure. So we
                    # can't test the bbmod_vals directy (inv-cov estimates
                    # in the paper) testing that the log-likelihood hasn't
                    # exploded is a reasonable first approximation.
                    cond2 = (ll > -1e15) & (ll < 1e15)
                    cond3 = (tau < Nt - buff) & (tau > buff)
                    cond = cond1 & cond2 & cond3
                }

                else{
                cond = FALSE
                }

                if(cond){
                    t_ll_l = c(t_ll_l, ll0, ll1)
                    t_cp_l = c(t_cp_l, tau + cp_l[i], cp_l[i + 1])
                    t_state_l = c(t_state_l, 1, 1)
                    t_bbmod_vals_base = c(t_bbmod_vals_base,
                                          list(bbmod_vals[[1]],
                                               bbmod_vals[[2]]))
                }

                else{
                    t_ll_l = c(t_ll_l, ll_l[i])
                    t_cp_l = c(t_cp_l, cp_l[i + 1])
                    t_state_l = c(t_state_l, 0)
                    t_bbmod_vals_base = c(t_bbmod_vals_base,
                                          list(bbmod_vals_base[[i]]))
                }
            }
            else {
                t_ll_l = c(t_ll_l, ll_l[i])
                t_cp_l = c(t_cp_l, cp_l[i + 1])
                t_state_l = c(t_state_l, 0)
                t_bbmod_vals_base = c(t_bbmod_vals_base,
                                      list(bbmod_vals_base[[i]]))
            }
        }
        ll_l = t_ll_l
        cp_l = t_cp_l
        state_l = t_state_l
        bbmod_vals_base = t_bbmod_vals_base
        print(paste("Current Changepoints:", cp_l))
    }
    res = list()
    res$tau_l = cp_l
    res$bbmod_vals = bbmod_vals_base
    return(res)
}
