simulated_annealing <- function(data, bbmod_vals, bbmod_params, bbmod_method, bbmod_ll, niter, min_beta, buff){
    """simulated annealing method for estimating chagne-points, returns the index of
    the estimated change-point

    Parameters
    ----------

        niter : int
            number of simulated annealing iterations
        min_beta : double
            minimum temperature to reach
        buff : int
            distance from edge to maintain for search

    Returns
    -------
        int index of change-point location
    """

    # TODO handle differently in class
    bbmod_vals = list()
    bbmod_vals[1] = bbmod_init_vals
    bbmod_vals[2] = bbmod_init_vals

    # initialize parameters
    N = dim(data)[1]
    tau = sample(1:N, 1)
    taup = tau
    iterations = 0
    beta = 1

    while((beta > min_beta) & (iterations < niter)){

        if(tau == taup){

            bbmod_val[1] = bbmod_method(data, tau, bbmod_val[1], bbmod_params)
            bbmod_val[2] = bbmod_method(data, tau, bbmod_val[2], bbmod_params)

            ll0 = bbmod_ll(data, tau, bbmod_val[1], bbmod_params)
            ll1 = bbmod_ll(data, tau, bbmod_val[2], bbmod_params)
            ll = ll0 + ll1
        }

        tau_p = sample(1:N, 1)
        ll0_p = bbmod_ll(data, tau, bbmod_val[1], bbmod_params)
        ll1_p = bbmod_ll(data, tau, bbmod_val[2], bbmod_params)
        ll_p = ll0_p + ll1_p

        prob = min(1, exp((ll_p - ll) / beta))

        u = runif(1)

        if (prob > u){
            tau = taup
        }

        beta = min_beta^(iterations/niter)
    }
    return(tau)
}


brute_force <- function(data, bbmod_init_vals, bbmod_params, bbmod_method, bbmod_ll, buff){
    """brute force method for estimating change-points, returns the index of
    the estimated change-point

    Parameters
    ----------
    buff : int
        distance from edge to maintain for search

    Returns
    -------
        int index of change-point location
    """

    # TODO handle differently in class
    bbmod_vals = list()
    bbmod_vals[1] = bbmod_init_vals
    bbmod_vals[2] = bbmod_init_vals

    N = dim(data)[1]
    ll_l = c()

    for(i in buff:(N-buff)){
        bbmod_val0 = bbmod_method(data, i, bbmod_val[0], bbmod_params)
        bbmod_val1 = bbmod_method(data, i, bbmod_val[1], bbmod_params)
        ll0 = bbmod_ll(data, i, bbmod_val0, bbmod_params)
        ll1 = bbmod_ll(data, i, bbmod_val1, bbmod_params)
        ll_l = c(ll_l, ll0 + ll1)
    }

    return(which.min(ll_l))
}


binary_segmentation <- function(data, bbmod_init_vals, bbmod_params, bbmod_method, bbmod_ll, method, thresh, buff){
    """handles the binary segmentation

    Parameters
    ----------
    method : int
        0 - simulated_annealing
        1 - ran_one
        2 - brute_force
    thresh : double
        threshold for ll diff to stop binary segmentation
    buff : int
        distance from edge to maintain for search

    Returns
    -------
    vector of estimated change-points
    """

    data_base = data
    bbmod_vals_base = list()
    bbmod_vals_base[1] = bbmod_init_vals

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
        t_bbmod_vals_base = c()

        for(i in 1:length(state_l)){

            if(state_l[i] == 1){

                data = data_base[cp_l[i]:cp_l[i+1],]
                Nt = dim(data)[1]

                if(Nt > 2 * (buff + 1)){

                    bbmod_vals = list()
                    bbmod_vals[1] = bbmod_init_vals
                    bbmod_vals[2] = bbmod_init_vals

                    if(method == 0){
                        tau = simulated_annealing()
                    }
                    if(method == 2){
                        tau = brute_force()
                    }

                    ll0 = bbmod_ll(data, tau, bbmod_vals[1], bbmod_params)
                    ll1 = bbmod_ll(data, tau, bbmod_vals[2], bbmod_params)
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
                    t_bbmod_vals_base = c(t_bbmod_vals, bbmod_vals[1],
                                          bbmod_vals[2])
                }

                else{
                    t_ll_l = c(t_ll_l, ll_l[i])
                    t_cp_l = c(t_cp_l, cp_l[i + 1])
                    t_state_l = c(t_state_l, 0)
                    t_bbmod_vals_base = c(t_bbmod_vals, bbmod_vals_
