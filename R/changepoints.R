.partition_data <- function(row, partition, tau){
    # TODO handle vector and matrices more cleanly
    dm = dim(row)
    N = dm[1]
    if(partition == 1){
        row = row[1:tau,]
    }
    else{
        row = row[(tau+1):N,]
    }
    return(row)
 }




#' @name changepointsMod-class
#'
#' @title An S4 class corresponding to the change-point model.
#'
#' @description An S4 class corresponding to the change-point model.
#'
#' @slot data A list containing the data for the change-point model.
#'       The exact structure of the data is dependent on the \code{bbmod}
#'       and \code{log_likelihood} provided.  In cases where the data is
#'       fairly simple, it should still be wrapped with a list, e.g.
#'       X = list(X), to allow changepointsMod to handle it properly.
#' @slot part_values A list containing the values estimated by
#'       \code{bbmod}. \code{part_values}, in particular, contain values
#'       that are updated independently for each partition (as opposed to
#'       \code{whole_values}).
#' @slot whole_values A list containing the values estimated by bbmod.
#'       whole values, in particular, contain values that are shared
#'       between partitions (as opposed to \code{part_values}).
#' @slot bbmod An R function for performing the black-box estimation.
#' @slot bbmod_params A list containing any additional parameters
#'       for \code{bbmod}.
#' @slot log_likelihood An R function for estimating the log-likelihood
#'       for the corresponding \code{bbmod}.
#' @slot ll_params A list containing any additional parameters for
#'       \code{log_likelihood}.
#' @slot trace A vector corresponding the the trace of the estimated
#'       change-points based on the method used.
#' @slot changepoints A scalar/vector corresponding to the changepoint(s)
#'       estimated based on the method used.
#' @slot mod_list A list corresponding to all the active single change-point
#'       models used with \code{binary_segmentation}.
#' @slot mod_range A list of the range of observations corresponding to each
#'       active model for \code{binary_segmentation}.
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname changepointsMod-class
#' @exportClass changepointsMod
changepointsMod <- setClass(
    # set class name
    "changepointsMod",

    # set slots
    slots = c(
            data = "list",
            part_values = "list",
            whole_values = "list",
            bbmod = "function",
            bbmod_params = "list",
            log_likelihood = "function",
            ll_params = "list",
            trace = "numeric",
            changepoints = "numeric",
            mod_list = "list",
            mod_range = "list"
            ),

    # confirm that starting values were provided for estimation
    # method
    validity=function(object)
    {
        if((length(object@part_values) == 0) &
           (length(object@whole_values) == 0)){
            return("Both part_values and whole_values were null.")
        }
        return(TRUE)
    }
)


#' @name bbmod_method
#'
#' @title Wrapper method for black-box estimation.
#'
#' @description Applies the black-box estimator to the specified partition given
#'              the current tau value.  Additionally, this wrapper handles the
#'              different data structures possible for \code{part_values} and
#'              \code{whole_values}.
#'
#' @param object Corresponding \code{changepointsMod} class.
#' @param part Index for current partition, should be 1 or 2.
#' @param tau Current change-point.  Should be between buff and N - buff.
#'
#' @return An updated version of the change-point model.  There are currently three
#'         possible updates depending on the form of the \code{part_values} and
#'         \code{whole_values} provided.  1) If only \code{part_values} are provided,
#'         then we assume the black-box method only updates \code{part_values.}
#'         2) If only \code{whole_values} are provide, we assume the black-box
#'         method only updates \code{whole_values}. 3) If both \code{part_values}
#'         and \code{whole_values} are provided, we assume that both are updated.
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname bbmod_method
setGeneric(name="bbmod_method",
           def=function(object, part, tau)
           {
               standardGeneric("bbmod_method")
           }
)

#' @rdname bbmod_method
setMethod(f="bbmod_method",
          signature="changepointsMod",
          definition=function(object, part, tau)
          {
              # extract data for current partition
              part_data = lapply(object@data,
                                 function(row) .partition_data(row, part, tau))
              print(dim(part_data[[1]]))

              # TODO add support for list values vs. list wrapped values
              if((length(object@part_values) > 0) &
                 (length(object@whole_values) == 0)){
                  params = c(part_data,
                             list(object@part_values[[part]]),
                             object@bbmod_params)
                  object@part_values[[part]] = do.call(object@bbmod, params)
              }
              else if((length(object@whole_values) > 0) &
                      (length(object@part_values) == 0)){
                  params = c(part_data,
                             list(object@whole_values),
                             object@bbmod_params)
                  object@whole_values = do.call(object@bbmod, params)
                  print("HELLO")
              }
              else if((length(object@whole_values) > 0) &
                      (length(object@part_values) > 0)){
                  group_values = list(list(object@part_values[[part]]),
                                      list(object@whole_values))
                  params = c(part_data,
                             group_values,
                             object@bbmod_params)
                  group_values = do.call(object@bbmod, params)
                  object@part_values[[part]] = group_values[[1]]
                  object@whole_values = group_values[[2]]
              }
              return(object)
          }
)


#' @name log_likelihood_method
#'
#' @title Wrapper method for log-likelihood estimation.
#'
#' @description Generates the log-likelihood for the specified partition given the
#'              current tau value.  Additionally, this wrapper handles the different
#'              data structures possible for part_values and whole_values.
#'
#' @param object Corresponding \code{changepointsMod} class.
#' @param part Index for current partition, should be 1 or 2.
#' @param tau Current change-point.  Should be between buff and N - buff.
#'
#' @return The log-likelihood estimate for the current state.  There are currently three
#'         possible versions depending on the form of the \code{part_values} and
#'         \code{whole_values} provided.  1) If only \code{part_values} are provided,
#'         then we assume the log-likelihood takes only the \code{part_values.}
#'         2) If only \code{whole_values} are provide, we assume the log-likelihood
#'         takes only the \code{whole_values}. 3) If both \code{part_values}
#'         and \code{whole_values} are provided, we assume that the log-likelihood
#'         takes both.
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname log_likelihood_method
setGeneric(name="log_likelihood_method",
           def=function(object, part, tau)
           {
               standardGeneric("log_likelihood_method")
           }
)

#' @rdname log_likelihood_method
setMethod(f="log_likelihood_method",
          signature="changepointsMod",
          definition=function(object, part, tau)
          {
              # extract data for current partition
              part_data = lapply(object@data,
                                 function(row) .partition_data(row, part, tau))

              if((length(object@part_values) > 0) &
                 (length(object@whole_values) == 0)){
                  params = c(part_data,
                             list(object@part_values[[part]]),
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
              else if((length(object@whole_values) > 0) &
                      (length(object@part_values) == 0)){
                  params = c(part_data,
                             list(object@whole_values),
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
              else if((length(object@whole_values) > 0) &
                      (length(object@part_values) > 0)){
                  group_values = list(list(object@part_values[[part]]),
                                      list(object@whole_values))
                  params = c(part_data,
                             group_values,
                             object@ll_params)
                  return(do.call(object@log_likelihood, params))
              }
          }
)


#' @name simulated_annealing
#'
#' @title Single change-point simulated annealing method
#'
#' @description Estimates a single change-point using the simulated annealing
#'              method.
#'
#' @param object Corresponding \code{changepointsMod} class.
#' @param niter Number of simulated annealing iterations.
#' @param min_beta Lowest temperature.
#' @param buff Distance from edge of sample to be maintained during search.
#'
#' @return An updated version of the change-point model.  The update will effect:
#'         1) the \code{part_values} and/or \code{whole_values} (depending on the initial
#'         values provided).  2) An estimate for the current change-point.  3) The trace
#'         for the search.
#'
#' @examples
#' set.seed(334)
#'
#' scp_data = read.table(system.file("extdata", "scp.txt", package="changepointsHD"))
#' scp_data = as.matrix(scp_data)
#'
#' # prox gradient black-box method
#' cov_est = cov(scp_data)
#' init = solve(cov_est)
#' res_map = prox_gradient_mapping(scp_data, init, 0.1, 0.99, 0.1, 100, 1e-20)
#'
#' # prox gradient black-box ll
#' res_ll = prox_gradient_ll(scp_data, res_map, 0.1)
#'
#' prox_gradient_params=list()
#' prox_gradient_params$update_w = 0.1
#' prox_gradient_params$update_change = 0.99
#' prox_gradient_params$regularizer = 0.1
#' prox_gradient_params$max_iter = 1
#' prox_gradient_params$tol = 1e-5
#'
#' prox_gradient_ll_params=list()
#' prox_gradient_ll_params$regularizer = 0.1
#'
#' changepoints_mod = changepointsMod(bbmod=prox_gradient_mapping,
#'                                  log_likelihood=prox_gradient_ll,
#'                                  bbmod_params=prox_gradient_params,
#'                                  ll_params=prox_gradient_ll_params,
#'                                  part_values=list(init, init),
#'                                  data=list(scp_data))
#' changepoints_mod = simulated_annealing(changepoints_mod, buff=10)
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname simulated_annealing
setGeneric(name="simulated_annealing",
           def=function(object, niter=500, min_beta=1e-4, buff=100)
           {
               standardGeneric("simulated_annealing")
           }
)

#' @rdname simulated_annealing
setMethod(f="simulated_annealing",
          signature="changepointsMod",
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
                  print("UPDATE1")
                  object = bbmod_method(object, 2, tau)
                  print("UPDATE2")
                  ll0 = log_likelihood_method(object, 1, tau)
                  print("LL1")
                  ll1 = log_likelihood_method(object, 2, tau)
                  print("LL2")
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


#' @name brute_force
#'
#' @title Single change-point brute force method.
#'
#' @description Estimates a single change-point by testing all possible
#'              change-points.
#'
#' @param object Corresponding \code{changepointsMod} class.
#' @param niter Number of iterations at each possible change-point.
#' @param buff Distance from edge of sample to be maintained during search.
#'
#' @return An updated version of the change-point model.  The update will effect:
#'         1) the \code{part_values} and/or \code{whole_values} (depending on the initial
#'         values provided).  2) An estimate for the current changepoint.  3) The trace
#'         for the search.
#'
#' @examples
#' set.seed(334)
#'
#' scp_data = read.table(system.file("extdata", "scp.txt", package="changepointsHD"))
#' scp_data = as.matrix(scp_data)
#'
#' # prox gradient black-box method
#' cov_est = cov(scp_data)
#' init = solve(cov_est)
#' res_map = prox_gradient_mapping(scp_data, init, 0.1, 0.99, 0.1, 100, 1e-20)
#'
#' # prox gradient black-box ll
#' res_ll = prox_gradient_ll(scp_data, res_map, 0.1)
#'
#' prox_gradient_params=list()
#' prox_gradient_params$update_w = 0.1
#' prox_gradient_params$update_change = 0.99
#' prox_gradient_params$regularizer = 0.1
#' prox_gradient_params$max_iter = 1
#' prox_gradient_params$tol = 1e-5
#'
#' prox_gradient_ll_params=list()
#' prox_gradient_ll_params$regularizer = 0.1
#'
#' changepoints_mod = changepointsMod(bbmod=prox_gradient_mapping,
#'                                  log_likelihood=prox_gradient_ll,
#'                                  bbmod_params=prox_gradient_params,
#'                                  ll_params=prox_gradient_ll_params,
#'                                  part_values=list(init, init),
#'                                  data=list(scp_data))
#' changepoints_mod = brute_force(changepoints_mod, buff=10)
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname brute_force
setGeneric(name="brute_force",
           def=function(object, niter=1, buff=100)
           {
               standardGeneric("brute_force")
           }
)

#' @rdname brute_force
setMethod(f="brute_force",
          signature="changepointsMod",
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


#' @name binary_segmentation
#'
#' @title Multiple change-point method.
#'
#' @description Estimates multiple change-points using the binary-segmentation
#'              method.  This does a breadth first search and uses the specified
#'              single change-point method for each sub-search.
#'
#' @param object Corresponding \code{changepointsMod} class.
#' @param method changepointHD method for finding single change-point.
#' @param thresh Stopping threshold for cost comparison.
#' @param buff Distance from edge of sample to be maintained during search.
#' @param method_params List of additional parameters for \code{method}.
#'
#' @return An updated version of the change-point model.  The update will effect:
#'         1) An estimate for the current set of change-points.  2) The \code{mod_list},
#'         this will correspond to all the active single change-point models
#'         generated during the binary-segmentation procedure.  Acitve models
#'         correspond to models that have not been superseded by more granular
#'         models.  3) The \code{mod_range}, this corresponds to the range of
#'         observations covered by each model.  It can be used to determine which
#'         models are active.
#'
#' @examples
#' set.seed(334)
#'
#' mcp_data = read.table(system.file("extdata", "mcp.txt", package="changepointsHD"))
#' mcp_data = as.matrix(mcp_data)
#'
#' # prox gradient black-box method
#' cov_est = cov(mcp_data)
#' init = solve(cov_est)
#' res_map = prox_gradient_mapping(mcp_data, init, 0.1, 0.99, 0.1, 100, 1e-20)
#'
#' # prox gradient black-box ll
#' res_ll = prox_gradient_ll(mcp_data, res_map, 0.1)
#'
#' prox_gradient_params=list()
#' prox_gradient_params$update_w = 0.1
#' prox_gradient_params$update_change = 0.99
#' prox_gradient_params$regularizer = 0.1
#' prox_gradient_params$max_iter = 1
#' prox_gradient_params$tol = 1e-5
#'
#' prox_gradient_ll_params=list()
#' prox_gradient_ll_params$regularizer = 0.1
#'
#' simulated_annealing_params = list()
#' simulated_annealing_params$buff=10
#'
#' changepoints_mod = changepointsMod(bbmod=prox_gradient_mapping,
#'                                  log_likelihood=prox_gradient_ll,
#'                                  bbmod_params=prox_gradient_params,
#'                                  ll_params=prox_gradient_ll_params,
#'                                  part_values=list(init, init),
#'                                  data=list(mcp_data))
#'
#' changepoints_mod = binary_segmentation(changepoints_mod, method=simulated_annealing,
#'                                        thresh=0, buff=10,
#'                                        method_params=simulated_annealing_params)
#'
#' @author \packageMaintainer{changepointsHD}
#'
#' @rdname binary_segmentation
setGeneric(name="binary_segmentation",
           def=function(object, method, thresh=0, buff=100,
                        method_params=list())
           {
               standardGeneric("binary_segmentation")
           }
)

#' @rdname binary_segmentation
setMethod(f="binary_segmentation",
          signature="changepointsMod",
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

                      tmod = changepointsMod(data=part_data,
                                            bbmod=object@bbmod,
                                            log_likelihood=object@log_likelihood,
                                            bbmod_params=object@bbmod_params,
                                            ll_params=object@ll_params,
                                            part_values=object@part_values,
                                            whole_values=object@whole_values)
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
              # then iterate over previous mod_list/mod_range and select
              # those models which are still active
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
              # join the previous active models with current active
              # models
              mod_list = c(n_mod_list, t_mod_list)
              mod_range = c(n_mod_range, t_mod_range)
          }
      object@mod_list = mod_list
      object@mod_range = mod_range
      object@changepoints = cp_l
      return(object)
      }
)
