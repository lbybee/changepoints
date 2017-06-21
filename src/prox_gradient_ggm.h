#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


// actual mapping function (bbmod_method)
mat prox_gradient_mapping(mat cov_est, mat theta_start, double update_w,
                          double update_change, double regularizer,
                          int max_iter, double tol);
// corresponding likelihood (bbmod_ll)
double prox_gradient_ll(mat data, mat theta_i, double regularizer);
