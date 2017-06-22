// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


// actual mapping function (bbmod_method)
arma::mat prox_gradient_mapping(arma::mat data, arma::mat theta_start,
                                double update_w, double update_change,
                                double regularizer, int max_iter, double tol);
// corresponding likelihood (bbmod_ll)
double prox_gradient_ll(arma::mat dataS, arma::mat theta_iS,
                        double regularizer);
