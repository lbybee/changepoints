#' @name simulated_annealing
#'
#' @title Estimates a single change-point using the simulated annealing
#'        method
#'
#' @description Estimates a single change-point using the simulated annealing
#'              method
#'
#' @param data matrix of actual data.
#' @param bbmod_init_vals initial estimates for arbitrary bbmod values
#' @param bbmod_method function corresponding to black box model itself
#' @param bbmod_ll function returning ll or comparable cost function for
#'                 black box model
#' @param niter number of simulated annealing iterations
#' @param min_beta lowest temperature
#' @param buff distance from edge of sample to be maintained
#' @param bbmod_method_params list of additional parameters for bbmod_method
#' @param bbmod_ll_params list of additional parameters for bbmod_ll
#'
#' @return List containing estimated change-point and thetas
#'
#' @author Leland Bybee \email{leland.bybee@@gmail.com}
simulated_annealing <- function(data, bbmod_init_vals, bbmod_method, bbmod_ll,
                                niter=500, min_beta=1e-4, buff=100,
                                bbmod_method_params=list(),
                                bbmod_ll_params=list()){
    "simulated annealing for esitmating change-points, returns a list
    containing the location of the estimated change-point as well as the
    estimated black box model values

    Paramters
    ---------

    data : matrix
        N x P matrix of actual data.  Note that given the way this is passed
        bbmod_method and bbmod_ll this could possibly be overloaded to handle
        more complex data structures.  The key is that we assume that there
        are N rows.  This is key because we must subset data for bbmod_method
        and bbmod_ll.
    bbmod_init_vals : arbitrary
        matrix of initial values for bbmod.  Again, this can also be
        overloaded.  Note that in the GGM case this is a P x P matrix for
        the theta values, but in the LDA case this can be both the
        topic-document proportions and the term-topic proportions (so a
        list of both).
    bbmod_method : function
        function used to estimate bbmod_vals
        Note that this should be set up to work as
        bbmod_method(data, init_vals, additional_parameters...)
    bbmod_ll : function
        function used to estimate the log likelihood for the bbmod
        Note that this should be set up to work as
        bbmod_ll(data, init_vals, additional_parameters...)
    niter : scalar
        number of simulated annealing iterations
    min_beta : scalar
        min beta to stop method when reached
    buff : scalar
        distance from edge of sample to be maintained for search
    bbmod_method_params : list
        any additional parameters that are to be passed to bbmod_method
    bbmod_ll_params : list
        any additional parameters that are to be passed to bbmod_ll

    Returns
    -------
    list
        tau : change-point
        bbmod_vals : list of bbmod_vals for each subset
    "

    # TODO add support for different init vals
    bbmod_vals = list(bbmod_init_vals, bbmod_init_vals)

    # initialize parameters
    N = dim(data)[1]
    tau = sample(buff:(N-buff), 1)
    taup = tau
    iterations = 0
    beta = 1

    while((beta > min_beta) & (iterations < niter)){

        if(tau == taup){

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
        }

        tau_p = sample(buff:(N-buff), 1)
        ll0_p = do.call(bbmod_ll, c(list(data[1:tau_p,], bbmod_vals[[1]]),
                                    bbmod_ll_params))
        ll1_p = do.call(bbmod_ll, c(list(data[(tau_p+1):N,], bbmod_vals[[2]]),
                                    bbmod_ll_params))
        ll_p = ll0_p + ll1_p

        prob = min(1, exp((ll_p - ll) / beta))

        u = runif(1)

        if (prob > u){
            tau = taup
        }

        beta = min_beta^(iterations/niter)
        iterations = iterations + 1
        print(iterations)
    }
    res = list()
    res$tau = tau
    res$bbmod_vals = bbmod_vals
    return(res)
}


#' @name brute_force
#'
#' @title Estimates a single change-point by searching all change-points
#'
#' @description Estimates a single change-point by search all change-points
#'
#' @param data matrix of actual data.
#' @param bbmod_init_vals initial estimates for arbitrary bbmod values
#' @param bbmod_method function corresponding to black box model itself
#' @param bbmod_ll function returning ll or comparable cost function for
#'                 black box model
#' @param buff distance from edge of sample to be maintained
#' @param bbmod_method_params list of additional parameters for bbmod_method
#' @param bbmod_ll_params list of additional parameters for bbmod_ll
#'
#' @return List containing estimated change-point and thetas
#'
#' @author Leland Bybee \email{leland.bybee@@gmail.com}

brute_force <- function(data, bbmod_init_vals, bbmod_method, bbmod_ll,
                        buff=100, bbmod_method_params=list(),
                        bbmod_ll_params=list()){
    "brute force method for estimating change-points, returns the index of
    the estimated change-point

    data : matrix
        N x P matrix of actual data.  Note that given the way this is passed
        bbmod_method and bbmod_ll this could possibly be overloaded to handle
        more complex data structures.  The key is that we assume that there
        are N rows.  This is key because we must subset data for bbmod_method
        and bbmod_ll.
    bbmod_init_vals : arbitrary
        matrix of initial values for bbmod.  Again, this can also be
        overloaded.  Note that in the GGM case this is a P x P matrix for
        the theta values, but in the LDA case this can be both the
        topic-document proportions and the term-topic proportions (so a
        list of both).
    bbmod_method : function
        function used to estimate bbmod_vals
        Note that this should be set up to work as
        bbmod_method(data, init_vals, additional_parameters...)
    bbmod_ll : function
        function used to estimate the log likelihood for the bbmod
        Note that this should be set up to work as
        bbmod_ll(data, init_vals, additional_parameters...)
    buff : scalar
        distance from edge of sample to be maintained for search
    bbmod_method_params : list
        any additional parameters that are to be passed to bbmod_method
    bbmod_ll_params : list
        any additional parameters that are to be passed to bbmod_ll

    Returns
    -------
    list
        tau : change-point
        bbmod_vals : list of bbmod_vals for each subset
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
        ll_l = c(ll_l, ll0 + ll1)
        print(i)
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
#' @title Estimates multiple change-points by binary segmentation
#'
#' @description Estimates multiple change-points by binary segmentations
#'              using provided single-change-point method to search each
#'              partition
#'
#' @param data matrix of actual data.
#' @param bbmod_init_vals initial estimates for arbitrary bbmod values
#' @param cp_method single change-point method used
#' @param bbmod_method function corresponding to black box model itself
#' @param bbmod_ll function returning ll or comparable cost function for
#'                 black box model
#' @param thresh stopping threshold for cost comparison
#' @param buff distance from edge of sample to be maintained
#' @param cp_method_params list of additional parameters for cp_method
#' @param bbmod_method_params list of additional parameters for bbmod_method
#' @param bbmod_ll_params list of additional parameters for bbmod_ll
#'
#' @return List containing estimated change-points and thetas
#'
#' @author Leland Bybee \email{leland.bybee@@gmail.com}
binary_segmentation <- function(data, bbmod_init_vals, cp_method,
                                bbmod_method, bbmod_ll, thresh=0, buff=100,
                                cp_method_params=list(),
                                bbmod_method_params=list(),
                                bbmod_ll_params=list()){
    "handles the binary segmentation

    Parameters
    ----------

    data : matrix
        N x P matrix of actual data.  Note that given the way this is passed
        bbmod_method and bbmod_ll this could possibly be overloaded to handle
        more complex data structures.  The key is that we assume that there
        are N rows.  This is key because we must subset data for bbmod_method
        and bbmod_ll.
    bbmod_init_vals : arbitrary
        matrix of initial values for bbmod.  Again, this can also be
        overloaded.  Note that in the GGM case this is a P x P matrix for
        the theta values, but in the LDA case this can be both the
        topic-document proportions and the term-topic proportions (so a
        list of both).
    cp_method : function
        which method should be used to estimate the individual change-points
        currently supports
            1. simulated annealing
            2. brute force
    bbmod_method : function
        function used to estimate bbmod_vals
        Note that this should be set up to work as
        bbmod_method(data, init_vals, additional_parameters...)
    bbmod_ll : function
        function used to estimate the log likelihood for the bbmod
        Note that this should be set up to work as
        bbmod_ll(data, init_vals, additional_parameters...)
    thresh : scalar
        threshold for likelihood comparison below which we no-longer accept
        change-points
    buff : scalar
        distance from edge of sample to be maintained for search
    bbmod_method_params : list
        any additional parameters that are to be passed to bbmod_method
    bbmod_ll_params : list
        any additional parameters that are to be passed to bbmod_ll

    Returns
    -------
    list
        tau_l : change-point vector
        bbmod_vals : list of bbmod_vals for each subset
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

                data = data[cp_l[i]:cp_l[i+1],]
                Nt = dim(data)[1]

                if(Nt > 2 * (buff + 1)){

                    datat = data[cp_l[i]:cp_l[i+1],]

                    tres = do.call(cp_method, list(datat, bbmod_init_vals),
                                                   bbmod_method, bbmod_ll,
                                                   cp_method_params,
                                                   bbmod_method_params,
                                                   bbmod_ll_params)
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
    }
    res = list()
    res$tau_l = cp_l
    res$bbmod_vals = bbmod_vals_base
    return(res)
}
