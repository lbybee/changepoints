#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
mat mapping(mat cov_est, mat theta_0,
            float update_w, float update_change,
            float regularizer, int max_iter, float tol){
    /* Produces the regularized estimation of the covariance matrix
     *
     * Parameters
     * ----------
     *
     *  cov_est : mat
     *      The covariance of the data for the current tau values, should
     *      be P x P
     *  theta_0 : mat
     *      The starting value for theta, should be P x P
     *  update_w : float
     *      The weight for the update to theta
     *  update_change : float
     *      The amount by which update_w shrinks with each failed run
     *  regularizer : float
     *      The threshold for the update
     *  max_iter : int
     *      Limits the maximum number iterations
     *  tol : float
     *      The tolerance at which updates are stopped
     *
     * Returns
     * -------
     *
     *  theta_p : mat
     *      current estimate for theta
     */

    int P = cov_est.n_cols;

    mat theta_p = eye<mat>(P, P);
    mat theta_k = theta_0;
    mat inv_theta = eye<mat>(P, P);

    int i = 0;

    float delta_norm = norm(theta_k - theta_p) / norm(theta_k);

    // NOTE may need to be a bool
    int state = 1;

    while (state and i < max_iter) {

        // TODO add error catching here
        inv_theta = inv(theta_k);
        theta_p = theta_k - update_w * (cov_est - inv_theta);
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < P; k++) {
                if (theta_p(j,k) <= -regularizer) {
                    theta_p(j,k) += regularizer;
                }
                else if (theta_p(j,k) >= regularizer) {
                    theta_p(j,k) -= regularizer;
                }
                else {
                    theta_p(j,k) = 0;
                }
            }
        }
        delta_norm = norm(theta_k - theta_p) / norm(theta_k);
        if (delta_norm < tol) {
            state = 0;
        }
        i += 1;
    }

    return theta_p;
}


// [[Rcpp::export]]
float log_likelihood(mat data, mat theta0, mat theta1, int tau,
                     float regularizer) {
    /* Generates the log-likelihood for the specified theta and
     * tau values
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix containing the data used for estimation
     *  theta0 : mat
     *      P x P inverse cov estimate for first partition
     *  theta1 : mat
     *      P x P inverse cov estimate for second partition
     *  tau : int
     *      Current changepoint estimate
     *  regularizer : float
     *      regularizing constant
     *
     * Returns
     * -------
     *
     *  scalar estimate for the log-likelihood
     */

    int N = data.n_rows;
    int P = data.n_cols;

    mat S0 = cov(data.rows(0, tau-1));
    mat S1 = cov(data.rows(tau, N-1));

    mat TdS0 = theta0 % S0;
    mat TdS1 = theta1 % S1;
    float tr_TdS0 = trace(TdS0);
    float tr_TdS1 = trace(TdS1);

    double val;
    double sign;
    float det0 = log_det(val, sign, theta0);
    float det1 = log_det(val, sign, theta1);

    float ll = tau * 0.5 * (- det0 + tr_TdS0);
    ll += regularizer * sqrt(log(P) / tau) * norm(theta0, 1);
    ll += (N - tau) * 0.5 * (- det1 + tr_TdS1);
    ll += regularizer * sqrt(log(P) / (N - tau)) * norm(theta1, 1);
    ll *= -1;

    int test = randi(1, distr_param(0, 10))(0);

    printf("%d", test);

    return ll;
}


// [[Rcpp::export]]
int proposal_uniform(int N, int buff){
    /* produces a possible tau value from T sampled uniformly
     *
     * Parameters
     * ----------
     *  N : int
     *      sample size, T
     *  buff : int
     *      buffer from edges to prevent issues
     *
     * Returns
     * -------
     *
     *  proposed tau value
     */

    int tau = randi(1, distr_param(buff, N - buff))(0);
    return tau;
}


// [[Rcpp::export]]
int proposal_gaussian(int tau, int N, int buff, float sigma2){
    /* produces a possible tau value from T using gaussian proposal
     *
     * Parameters
     * ----------
     *  tau : int
     *      current tau value
     *  N : int
     *      sample size, T
     *  buff : int
     *      buffer from edges to prevent issues
     *  sigma2 : float
     *      variance for proposal
     *
     * Returns
     * -------
     *
     *  proposed tau value
     */

     return tau;
}


// [[Rcpp::export]]
int proposal_mixture(int tau, int N, int buff, float sigma2, float beta){
    /* produces a possible tau value from T using mixture proposal
     *
     * Parameters
     * ----------
     *  tau : int
     *      current tau value
     *  N : int
     *      sample size, T
     *  buff : int
     *      buffer from edges to prevent issues
     *  sigma2 : float
     *      variance for proposal
     *  beta : float
     *      mixture proporitions
     *
     * Returns
     * -------
     *
     *  proposed tau value
     */

     return tau;
}


// [[Rcpp::export]]
float simulated_annealing(mat data, int tau=-1, int buff=10,
                          float regularizer=1., float update_w=1.,
                          float update_change=0.9, int mapping_iter=1,
                          int max_iter=500, int cooling=0, int kernel=0,
                          float beta_min=0.01, float d=1.1, float tol=0.00001){
    /* estimates a single changepoint using the simulated annealing
     * algorithm
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix containing the data used for estimation
     *  tau : int
     *      starting tau, if -1 will be drawn randomly
     *  buff : int
     *      n_0, how far to say from the edges
     *  regularizer : float
     *      regularizing constant, see mapping
     *  update_w : float
     *      step size, see mapping
     *  update_change : float
     *      change to update_w, see mapping
     *  mapping_iter : int
     *      maximum number of iterations for mapping
     *  max_iter : int
     *      maximum number of iterations for simulated annealing
     *  cooling : int
     *      which cooling schedule to use
     *      - 0 : exp
     *      - 1 : log
     *      - 2 : inverse linear
     *      - 3 : no cooling
     *  kernel : int
     *      which proposal kernel to use
     *      - 0 : uniform
     *      - 1 : gaussian
     *      - 2 : mixture
     *  beta_min : float
     *      minimum temperature at which to stop
     *  d : float
     *      inialitization parameter for temp
     *  tol : float
     *      tolerance for stopping mapping
     *
     * Returns
     * -------
     *
     *  esitmated tau value
     */

    int N = data.n_rows;
    int P = data.n_cols;
    int c;
    int beta;
    float temp_regularizer;

    float llp;
    float prob;
    float u;

    if (cooling == 1){
        c = log(d);
    }
    else if (cooling == 2){
        d = (max_iter * beta_min) / (1 - beta_min);
        c = d;
    }

    if (tau == -1){
        tau = randi(1, distr_param(buff, N - buff))(0);
    }

    int iterations = 0;
    int taup = -1;

    mat S_inv = inv(cov(data));

    mat theta0 = S_inv;
    mat theta1 = S_inv;

    mat S0 = cov(data.rows(0, tau-1));
    mat S1 = cov(data.rows(tau, N-1));
    temp_regularizer = regularizer * sqrt(log(P) / tau);
    theta0 = mapping(S0, theta0, update_w, update_change, temp_regularizer,
                     max_iter, tol);
    temp_regularizer = regularizer * sqrt(log(P) / (N - tau));
    theta1 = mapping(S1, theta1, update_w, update_change, temp_regularizer,
                     max_iter, tol);

    float ll = log_likelihood(data, theta0, theta1, tau, regularizer);

    if ( cooling == 0){
        beta = pow(beta_min, (0./max_iter));
    }
    else if ( cooling == 1){
        beta = c / log(d);
    }
    else if ( cooling == 2){
        beta = c / d;
    }
    else if ( cooling == 3){
        beta = 1.;
    }

    while (beta > beta_min and iterations < max_iter){

        if (tau == taup){

            S0 = cov(data.rows(0, tau-1));
            S1 = cov(data.rows(tau, N-1));
            temp_regularizer = regularizer * sqrt(log(P) / tau);
            theta0 = mapping(S0, theta0, update_w, update_change,
                             temp_regularizer, max_iter, tol);
            temp_regularizer = regularizer * sqrt(log(P) / (N - tau));
            theta1 = mapping(S1, theta1, update_w, update_change,
                             temp_regularizer, max_iter, tol);

        }

        if ( kernel == 0){
            taup = proposal_uniform(N, buff);
        }
        else if (kernel == 1){
            taup = proposal_gaussian(tau, N, buff, N);
        }
        else if (kernel == 2){
            taup = proposal_mixture(tau, N, buff, N, beta);
        }

        llp = log_likelihood(data, theta0, theta1, taup, regularizer);

        prob = fmin(exp((llp - ll) / beta), 1);

        vec v = randu(1);
        u = v(0);
        if (prob > u){
            tau = taup;
        }

        iterations += 1;

        if ( cooling == 0){
            beta = pow(beta_min, (iterations/max_iter));
        }
        else if ( cooling == 1){
            beta = c / log(d + iterations);
        }
        else if ( cooling == 2){
            beta = c / (d + iterations);
        }
    }
    return tau;
}
