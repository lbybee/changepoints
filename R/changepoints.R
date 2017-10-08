partition_data <- function(row, partition, tau){
    N = dim(row)[1]
    if(partition == 1){
        row = row[1:tau,]
    }
    else{
        row = row[(tau+1):N,]
    }
    return(row)
 }




changepointMod <- setClass(
    # set class name
    "changepointMod",

    # set slots
    slots = c(
            ll_params = "list",
            bbmod_params = "list",
            data = "list",
            part_values = "list",
            general_values = "list",
            bbmod = "function",
            log_likelihood = "function",
            trace = "numeric",
            changepoints = "numeric",
            mod_list = "list",
            mod_range = "list"
            ),

    # set default values
    prototype = list(
                ll_params = list(),
                bbmod_params = list(),
                data = list(),
                part_values = list(),
                general_values = list()
                ),

    # confirm that only one of part_values and general_values
    # was provided.  Currently, we don't support models that
    # require both
    validity=function(object)
    {
        if((length(object@part_values) > 0) &
           (length(object@general_values) > 0)){
            return("Both part_values and general_values were provided.")
        }
        if((length(object@part_values) == 0) &
           (length(object@general_values) == 0)){
            return("Both part_values and general_values were null.")
        }
        return(TRUE)
    }
)


setGeneric(name="bbmod_method",
           def=function(object, part, tau)
           {
               standardGeneric("bbmod_method")
           }
)

setMethod(f="bbmod_method",
          signature="changepointMod",
          definition=function(object, part, tau)
          {
              # extract data for current partition
              part_data = lapply(object@data,
                                 function(row) partition_data(row, part, tau))

              if((length(object@part_values) > 0) &
                 (length(object@general_values) == 0)){
                  params = c(part_data,
                             list(object@part_values[[part]]),
                             object@bbmod_params)
                  object@part_values[[part]] = do.call(object@bbmod, params)
              }
              else if((length(object@general_values) > 0) &
                      (length(object@part_values) == 0)){
                  params = c(part_data,
                             object@general_values,
                             object@bbmod_params)
                  object@general_values = do.call(object@bbmod, params)
              }
              else if((length(object@general_values) > 0) &
                      (length(object@part_values) > 0)){
                  group_values = list(list(object@part_values[[part]]),
                                      object@general_values)
                  params = c(part_data,
                             group_values,
                             object@bbmod_params)
                  group_values = do.call(object@bbmod, params)
                  object@part_values[[part]] = group_values[[1]]
                  object@general_values = group_values[[2]]
              }
              return(object)
          }
)


setGeneric(name="log_likelihood_method",
           def=function(object, part, tau)
           {
               standardGeneric("log_likelihood_method")
           }
)

setMethod(f="log_likelihood_method",
          signature="changepointMod",
          definition=function(object, part, tau)
          {
              # extract data for current partition
              part_data = lapply(object@data,
                                 function(row) partition_data(row, part, tau))

              if((length(object@part_values) > 0) &
                 (length(object@general_values) == 0)){
                  params = c(part_data,
                             list(object@part_values[[part]]),
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
              else if((length(object@general_values) > 0) &
                      (length(object@part_values) == 0)){
                  params = c(part_data,
                             object@general_values,
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
              else if((length(object@general_values) > 0) &
                      (length(object@part_values) > 0)){
                  group_values = list(list(object@part_values[[part]]),
                                      object@general_values)
                  params = c(part_data,
                             group_values,
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
          }
)


setGeneric(name="simulated_annealing",
           def=function(object, niter=500, min_beta=1e-4, buff=100)
           {
               standardGeneric("simulated_annealing")
           }
)

setMethod(f="simulated_annealing",
          signature="changepointMod",
          definition=function(object, niter, min_beta, buff)
          {
          # TODO might be a cleaner way to handle N
          N = dim(object@data[[1]])[1]
          ptaus = buff:(N-buff)
          tau = sample(ptaus, 1)
          taup = tau
          iterations = 0
          beta = 1
          change = TRUE
          trace = c()

          while((beta > min_beta) & (iterations < niter)){
              if(change){
                  object = bbmod_method(object, 1, tau)
                  object = bbmod_method(object, 2, tau)
                  ll0 = log_likelihood_method(object, 1, tau)
                  ll1 = log_likelihood_method(object, 2, tau)
                  ll = ll0 + ll1
                  change = FALSE
              }
              taup = sample(ptaus[-which(ptaus == tau)], 1)
              ll0p = log_likelihood_method(object, 1, taup)
              ll1p = log_likelihood_method(object, 2, taup)
              llp = ll0p + ll1p
              prob = min(1, exp((llp - ll) / beta))
              u = runif(1)
              if(prob > u){
                  tau = taup
                  change = TRUE
              }
              trace = c(trace, tau)
              beta = min_beta^(iterations/niter)
              iterations = iterations + 1
          }
          object@trace = trace
          object@changepoints = c(tau)
          return(object)
          }
)


setGeneric(name="brute_force",
           def=function(object, niter=1, buff=100)
           {
               standardGeneric("brute_force")
           }
)

setMethod(f="brute_force",
          signature="changepointMod",
          definition=function(object, niter, buff)
          {
          N = dim(object@data[[1]])[1]
          ll_l = c()
          trace = c()

          for(tau in buff:(N-buff)){
              for(iteration in 1:niter){
                  object = bbmod_method(object, 1, tau)
                  object = bbmod_method(object, 2, tau)
              }
              ll0 = log_likelihood_method(object, 1, tau)
              ll1 = log_likelihood_method(object, 2, tau)
              ll = ll0 + ll1
              ll_l = c(ll_l, ll)
              trace = c(trace, tau)
          }
          tau = which.min(ll_l)
          for(iteration in 1:niter){
              object = bbmod_method(object, 1, tau)
              object = bbmod_method(object, 2, tau)
          }
          object@trace = trace
          object@changepoints = c(tau)
          return(object)
          }
)


setGeneric(name="binary_segmentation",
           def=function(object, method, thresh=0, buff=100,
                        method_params=list())
           {
               standardGeneric("binary_segmentation")
           }
)

setMethod(f="binary_segmentation",
          signature="changepointMod",
          definition=function(object, method, thresh, buff, method_params)
          {
          N = dim(object@data[[1]])[1]

          cp_l = c(0, N)
          ll_l = c(-1e20)
          state_l = c(1)

          mod_list = list(NULL)
          mod_range = list(c(0, N))

          while(sum(state_l) > 0){

              t_cp_l = c(0)
              t_ll_l = c()
              t_state_l = c()
              t_mod_list = list()
              t_mod_range = list()

              for(i in 1:length(state_l)){

                  if(state_l[i] == 1){
                      part_data = lapply(object@data,
                                         function(x) x[cp_l[i]:cp_l[i+1],])
                      Nt = dim(part_data[[1]])[1]

                      if(Nt > 2 * (buff + 1)){

                      tmod = changepointMod(data=part_data,
                                            bbmod=object@bbmod,
                                            log_likelihood=object@log_likelihood,
                                            bbmod_params=object@bbmod_params,
                                            ll_params=object@ll_params,
                                            part_values=object@part_values,
                                            general_values=object@general_values)
                      tmod = do.call(method, c(list(tmod), method_params))
                      tau = tmod@changepoints
                      ll0 = log_likelihood_method(tmod, 1, tau)
                      ll1 = log_likelihood_method(tmod, 2, tau)
                      ll = ll0 + ll1
                      cond1 = (ll - ll_l[i]) > thresh
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
                          t_mod_list = c(t_mod_list, list(tmod))
                          t_mod_range = c(t_mod_range,
                                          list(c(cp_l[i], cp_l[i + 1])))
                      }

                      else{
                          t_ll_l = c(t_ll_l, ll_l[i])
                          t_cp_l = c(t_cp_l, cp_l[i + 1])
                          t_state_l = c(t_state_l, 0)

                      }
                  }
                  else {
                      t_ll_l = c(t_ll_l, ll_l[i])
                      t_cp_l = c(t_cp_l, cp_l[i + 1])
                      t_state_l = c(t_state_l, 0)

                  }
              }
              ll_l = t_ll_l
              cp_l = t_cp_l
              state_l = t_state_l

              # here we have to update the mod list with only those mods
              # that haven't been excluded.

              # first build index of all data points in current mods
              ind = c()
              for(cov in t_mod_range){
                  ind = c(ind, cov[1]:cov[2])
              }
              n_mod_list = list()
              n_mod_range = list()
              n_ind = 1
              for(mod_ind in 1:length(mod_range)){
                  cov = mod_range[[mod_ind]]
                  tind = cov[1]:cov[2]
                  tind = sum(!(tind %in% ind))
                  if(tind > 0){
                      n_mod_list[[n_ind]] = mod_list[[mod_ind]]
                      n_mod_range[[n_ind]] = mod_range[[mod_ind]]
                      n_ind = n_ind + 1
                  }
              }
              mod_list = c(n_mod_list, t_mod_list)
              mod_range = c(n_mod_range, t_mod_range)
          }
      object@mod_list = mod_list
      object@mod_range = mod_range
      object@changepoints = cp_l
      return(object)
      }
)
