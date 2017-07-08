// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "prox_gradient_ggm.h"

using namespace std;
using namespace Rcpp;
using namespace arma;


// proximal gradient black box model
//' @name prox_gradient_mapping
//'
//' @title Proximal-gradient mapping method.
//'
//' @description Performs the proximal-gradient mapping operation to
//'              estimate a regularized version of the inverse cov.
//'              matrix.  Follows the procedure described in,
//'              http://dept.stat.lsa.umich.edu/~yvesa/sto_prox.pdf
//'
//' @param data N x P matrix corresponding to the raw data.
//' @param theta_start Initial value for precision estimate.
//' @param update_w Step size for prox-gradient mapping.
//' @param update_change Proportion of \code{update_w} to keep when
//'        the algorithm fails to successfully estimate precision.
//' @param regularizer Regularizing constant, lambda.
//' @param max_iter Number of mapping iterations.
//' @param tol Tolerance at which the algorithm stops running.
//'
//' @return Theta (precision matrix) estimate.
//'
//' @author \packageMaintainer{changepointsHD}
// [[Rcpp::export]]
arma::mat prox_gradient_mapping(arma::mat data, arma::mat theta_start,
                                double update_w, double update_change,
                                double regularizer, int max_iter, double tol){
    /* Produces the regularized estimation of the covariance matrix using
     * the proximal gradient procedure described in
     *
     * http://dept.stat.lsa.umich.edu/~yvesa/sto_prox.pdf
     */

    int N = data.n_rows;
    int P = data.n_cols;

    // update the regularizer to reflect data subset
    regularizer *= sqrt(log(P) / N);

    mat cov_est = cov(data);

    // TODO may not need to fill theta_p and inv_theta with 0s
    // proposed theta estimate
    mat theta_p = eye<mat>(P, P);
    // current theta estimate
    mat theta_k = theta_start;
    mat inv_theta = eye<mat>(P, P);

    int i = 0;

    float delta_norm = norm(theta_k - theta_p) / norm(theta_k);

    bool state = true;

    while (state and i < max_iter) {

        try {
            inv_theta = inv(theta_k);
            theta_p = theta_k - update_w * (cov_est - inv_theta);
            for (int j = 0; j < P; j++) {
                // first we handle the diagonals
                if (theta_p(j,j) <= -regularizer * update_w) {
                    theta_p(j,j) += regularizer * update_w;
                }
                else if (theta_p(j,j) >= regularizer * update_w) {
                    theta_p(j,j) -= regularizer * update_w;
                }
                else {
                    theta_p(j,j) = 0;
                }
                // now we off diagonals.  Note that we only need
                // to check the upper triangle because the matrix
                // is symmetric
                for (int k = j + 1; k < P; k++) {
                    if (theta_p(j,k) <= -regularizer * update_w) {
                        theta_p(j,k) += regularizer * update_w;
                        theta_p(k,j) += regularizer * update_w;
                    }
                    else if (theta_p(j,k) >= regularizer * update_w) {
                        theta_p(j,k) -= regularizer * update_w;
                        theta_p(k,j) -= regularizer * update_w;
                    }
                    else {
                        theta_p(j,k) = 0;
                        theta_p(k,j) = 0;
                    }
                }
            }
        }
        // TODO add correct error here
        catch(...) {
            update_w *= update_change;
            theta_k = theta_start;
        }
        delta_norm = norm(theta_k - theta_p) / norm(theta_k);
        theta_k = theta_p;
        if (delta_norm < tol) {
            state = false;
        }
        i += 1;
    }

    return(theta_p);
}


// proximal gradient black box model log likelihood
//' @name prox_gradient_ll
//'
//' @title Proxmal-gradient log-likelihood estimator.
//'
//' @description Estimates the log-likeihood for the corresponding
//'              precision matrix and data set.
//'
//' @param data N x P matrix corresponding to the raw data.
//' @param theta_i Estimate for precision.
//' @param regularizer Regularizing constant, lambda.
//'
//' @return Log-likelihood estimate.
//'
//' @author \packageMaintainer{changepointsHD}
// [[Rcpp::export]]
double prox_gradient_ll(arma::mat data, arma::mat theta_i,
                        double regularizer) {
    /* Generates the log-likelihood for the specified theta and
     * tau values
     *
     * Notes
     * -----
     *
     *  - {N_tau / 2 * [tr(theta_i.T * S_i) - logdet(theta_i)]
     *     + lambda_tau * || theta_i || / 2}
     *  We take the negative here because we are minimizing
     *  this. TODO confirm that everything lines up
     *
     *  Also, we divide the regularizer by 2 so that we can
     *  include one in both likelihood (and not have to
     *  have this as external code).
     */

    int N = data.n_rows;
    int P = data.n_cols;

    mat S = cov(data);
    mat TdS = theta_i * S;
    double tr_TdS = trace(TdS);

    double sign;
    double det; log_det(det, sign, theta_i);

    double ll = N * 0.5 * (-det + tr_TdS);
    ll += regularizer * sqrt(log(P) / (N)) * norm(theta_i, 1) * 0.5;

    return -ll;
}
