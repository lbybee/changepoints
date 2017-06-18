#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


// changepoint model class
class ChangepointModel {
    public:
        // ---- parameters and variables ----
        // data
        mat data; // the data used for estimating the model, N x K

        // method params
        List method_params; // the different methods require different params
                            // and these are contained here

        // changepoint estimation params
        vec changepoints; // arbitrary length vector of changepoints, includes
                          // end points (0, N)

        // bbmod specific vars
        List bbmod_est; // arbitrary List containing black box model estimates
        List bbmod_params; // black box model specific params

        // bbmod functions
        Function bbmod_method; // Function used to generate bbmod_est
        Function bbmod_ll; // Function used to estimate bbmod log likelihood

        // ------------ functions ------------
        void init(mat dataS, List bbmod_params,
        int simulated_annealing();
        int brute_force();
        int rank_one();
        vec binary_segmentation();
