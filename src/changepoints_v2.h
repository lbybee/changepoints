#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;

// Note that the mapping and gg_loglikelihood functions are included
// primarily for testing purposes and to provide an example use case
//
// mapping function
mat mapping(mat cov_est, mat theta_start, double update_w,
            double update_change, double regularizer, int max_iter,
            double tol);
// gaussian graphical model log-likelihood
double gg_loglikelihood(mat data, mat theta_i, double regularizer);


// changepoint model class
class ChangepointModel {
    public:
        // ---- parameters and variables ----
        // data
        mat data; // the data used for estimating the model, N x K

        // bbmod specific vars
        List bbmod_est; // arbitrary List containing black box model estimates
        List bbmod_params; // black box model specific params
        List bbmod_ll_params; // black box model log-likelihood params

        // bbmod functions
        // TODO we should ultimately overload these so people can use native
        // Cpp functions as well
        //
        // Currently both of these should return Lists TODO change this
        Function bbmod_method; // Function used to generate bbmod_est
        Function bbmod_ll; // Function used to estimate bbmod log likelihood

        // ------------ functions ------------
        void init(mat dataS, List bbmod_paramsS, List bbmod_ll_params,
                  Function bbmod_methodS, bbmod_llS)
        int simulated_annealing(int niter=500);
        int brute_force(int niter=500);
        int rank_one(int niter=500);
        vec binary_segmentation(int method_ind, List method_params,
                                double threshold);
}
