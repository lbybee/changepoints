#include "prox_gradient_ggm.h"
#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


List log_likelihood_rank_one(mat data, mat S0, mat S1, mat theta0,
                             mat theta1, int buff, int tau,
                             float regularizer){
    /* Handles the log-likelihood estimation using the rank one method
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix of data used for estimation
     *  S0/S1 : mat
     *      P x P covariance estimates
     *  theta0/theta1 : mat
     *      P x P current inv cov estimates
     *  buff : int
     *      n_0, buffer to keep from edges
     *  tau : int
     *      current changepoint estimate
     *  regularizer : float
     *      regularizing constant
     *
     * Returns
     * -------
     *
     * estimate for changepoint and ll
     */

    int N = data.n_rows;
    int P = data.n_cols;

    mat TdS0 = theta0 % S0;
    mat TdS1 = theta1 % S1;

    float tr_TdS0 = trace(TdS0);
    float tr_TdS1 = trace(TdS1);

    double sign;
    double det0; log_det(det0, sign, theta0);
    double det1; log_det(det1, sign, theta1);

    float ll = tau * 0.5 * (-det0 + tr_TdS0);
    ll += regularizer * sqrt(log(P) / tau) * norm(theta0, 1);
    ll += (N - tau) * 0.5 * (-det1 + tr_TdS1);
    ll += regularizer * sqrt(log(P) / (N - tau)) * norm(theta1, 1);
    ll *= -1;

    vec ll_l = zeros(N - 2 * buff + 1);

    ll_l(tau - buff) = ll;

    mat Sp0 = S0;
    mat Sp1 = S1;

    mat TdSp0;
    mat TdSp1;

    float tr_TdSp0;
    float tr_TdSp1;

    for(int i = tau - 1; i >= buff; --i){

        mat op_data = zeros(1, P);
        op_data.row(0) = data.row(i);
        mat rank_one_update = kron(trans(op_data), op_data);

        Sp0 = (i * Sp0 - rank_one_update) / (i - 1);
        Sp1 = ((N - i - 1) * Sp1 + rank_one_update) / (N - i);

        TdSp0 = theta0 % Sp0;
        TdSp1 = theta1 % Sp1;

        tr_TdSp0 = trace(TdSp0);
        tr_TdSp1 = trace(TdSp1);

        ll = i * 0.5 * (-det0 + tr_TdSp0);
        ll += regularizer * sqrt(log(P) / i) * norm(theta0, 1);
        ll += (N - i) * 0.5 * (-det1 + tr_TdSp1);
        ll += regularizer * sqrt(log(P) / (N - i)) * norm(theta1, 1);
        ll *= -1;

        ll_l(i - buff) = ll;

    }

    Sp0 = S0;
    Sp1 = S1;

    for(int i = tau + 1; i <= N - buff; i++){

        mat op_data = zeros(1, P);
        op_data.row(0) = data.row(i);
        mat rank_one_update = kron(trans(op_data), op_data);

        Sp0 = ((i - 1) * Sp0 - rank_one_update) / i;
        Sp1 = ((N - i) * Sp1 + rank_one_update) / (N - i - 1);

        TdSp0 = theta0 % Sp0;
        TdSp1 = theta1 % Sp1;

        tr_TdSp0 = trace(TdSp0);
        tr_TdSp1 = trace(TdSp1);

        ll = i * 0.5 * (-det0 + tr_TdSp0);
        ll += regularizer * sqrt(log(P) / i) * norm(theta0, 1);
        ll += (N - i) * 0.5 * (-det1 + tr_TdSp1);
        ll += regularizer * sqrt(log(P) / (N - i)) * norm(theta1, 1);
        ll *= -1;
        ll_l(i - buff) = ll;

    }

    int min_tau = -1;
    // TODO fix this
    float ll_min = -1e20;
    for (int i = 0; i <= N - 2 * buff; i++){
        if (ll_l(i) > ll_min){
            min_tau = i + buff;
            ll_min = ll_l(i);
        }
    }

    TdS0 = theta0 % cov(data.rows(0, min_tau-1));
    TdS1 = theta1 % cov(data.rows(min_tau, N - 1));

    tr_TdS0 = trace(TdS0);
    tr_TdS1 = trace(TdS1);

    float ll0 = min_tau * 0.5 * (- det0 + tr_TdS0);
    ll0 += regularizer * sqrt(log(P) / min_tau) * norm(theta0, 1);
    float ll1 = (N - min_tau) * 0.5 * (- det1 + tr_TdS1);
    ll1 += regularizer * sqrt(log(P) / (N - min_tau)) * norm(theta1, 1);
    float ll_mod = -(ll0 + ll1);

    List res;
    res("tau") = min_tau;
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;

    return res;
}

// Rank one change-point estimation procedure
//' @name rank_one
//'
//' @title Rank one update single change-point estimation.
//'
//' @description This is a method for estimating a single-changepoint
//'              which takes advantage of the special structure
//'              of the Gaussian graphical model.  It cannot take
//'              arbitrary black-box models like \code{simulated_annealing}
//'              or \code{brute_force}.  However, it can still be run within
//'              \code{binary_segmentation}.
//'
//' @param data N x P Matrix corresponding to the raw data.
//' @param theta_init Initial value for theta estimate.
//' @param buff Distance to maintain from edge of sample.
//' @param regularizer Regularizing constant, lambda.
//' @param tau Initial Estimate for change-point.
//' @param max_iter Maximum number of rank-one updates to be
//'        run.
//' @param update_w Step size for prox-gradient.
//' @param update_change Proportion of \code{update_w} to keep when
//'        the algorithm fails to successfully estimate theta.
//' @param mapping_iter Number of mapping iterations.
//' @param tol Tolerance at which the algorithm stops running.
//'
//' @return List containing the estimated change-point and
//'         theta values.
//'
//' @author \packageMaintainer{changepointsHD}
// [[Rcpp::export]]
List rank_one(arma::mat data, arma::mat theta_init, int buff=10,
              float regularizer=1., int tau=-1, int max_iter=25,
              float update_w=1., float update_change=0.9,
              int mapping_iter=1, float tol=0.00001){
    /* Does the rank one estimateion of the changepoint
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix containing the data for estimateion, note data
     *      should be mean centered
     *  theta_init : mat
     *      Initial value of theta used to align rank_one with the
     *      other methods
     *  buff : int
     *      buffer to keep proposal from edge
     *  regularizer : float
     *      regularizing constant
     *  tau : int
     *      starting proposal for tau, if -1 is selected uniformly
     *  max_iter : int
     *      maximum number of rank one iterations
     *  update_w : float
     *      see mapping
     *  update_change : float
     *      see mapping
     *  mapping_iter : int
     *      see mapping
     *  tol : float
     *      see mapping
     *
     * Returns
     * -------
     *
     *  List containing change-point estimate and theta estimates
     */

    int N = data.n_rows;
    int P = data.n_cols;

    if (tau == -1){
        tau = randi(1, distr_param(buff, N - buff))(0);
    }

    int iterations = 0;

//    mat S_inv = inv(cov(data));

//    mat theta0 = S_inv;
//    mat theta1 = S_inv;
    mat theta0 = theta_init;
    mat theta1 = theta_init;

    mat S0;
    mat S1;
    float temp_regularizer;

    List res;
    List ll_res;

    float ll0;
    float ll1;
    float ll_mod;

    while (iterations < max_iter){

        mat data0 = data.rows(0, tau-1);
        mat data1 = data.rows(tau, N-1);

        // TODO this is generated both inside and outside prox_gradient
        // should be fixed cleanly
        S0 = cov(data0);
        S1 = cov(data1);

//        temp_regularizer = regularizer * sqrt(log(P) / tau);
        theta0 = prox_gradient_mapping(data0, theta0, update_w,
                                       update_change, temp_regularizer,
                                       mapping_iter, tol);
//        temp_regularizer = regularizer * sqrt(log(P) / (N - tau));
        theta1 = prox_gradient_mapping(data1, theta1, update_w,
                                       update_change, temp_regularizer,
                                       mapping_iter, tol);
        ll_res = log_likelihood_rank_one(data, S0, S1, theta0, theta1, buff,
                                         tau, regularizer);
        tau = ll_res("tau");
        ll0 = ll_res("ll0");
        ll1 = ll_res("ll1");
        ll_mod = ll_res("ll_mod");
        iterations += 1;
    }
    List bbmod_vals = List(2);
    bbmod_vals(0) = theta0;
    bbmod_vals(1) = theta1;
    res("tau") = tau;
    res("bbmod_vals") = bbmod_vals;

    return res;
}
