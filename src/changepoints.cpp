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
List log_likelihood(mat data, mat theta0, mat theta1, int tau,
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
     *  List containing ll for each parition as well as full ll
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

    float ll0 = tau * 0.5 * (- det0 + tr_TdS0);
    ll0 += regularizer * sqrt(log(P) / tau) * norm(theta0, 1);
    float ll1 = (N - tau) * 0.5 * (- det1 + tr_TdS1);
    ll1 += regularizer * sqrt(log(P) / (N - tau)) * norm(theta1, 1);
    float ll_mod = -(ll0 + ll1);

    List res;
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;

    return res;
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
List simulated_annealing(mat data, int tau=-1, int buff=10,
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
     *  a list containing the estimated tau, thetas, ll for partitions and
     *  full ll
     */

    int N = data.n_rows;
    int P = data.n_cols;
    int c;
    float beta = 1;
    float temp_regularizer;

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

    List ll_res = log_likelihood(data, theta0, theta1, tau, regularizer);
    float ll0 = ll_res("ll0");
    float ll1 = ll_res("ll1");
    float ll_mod = ll_res("ll_mod");
    float ll_modp;

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
            List ll_res = log_likelihood(data, theta0, theta1, tau, regularizer);
            ll0 = ll_res("ll0");
            ll1 = ll_res("ll1");
            ll_mod = ll_res("ll_mod");

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

        List ll_resp = log_likelihood(data, theta0, theta1, taup, regularizer);
        ll_modp = ll_resp("ll_mod");

        prob = fmin(exp((ll_modp - ll_mod) / beta), 1);

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

    List res;
    res("tau") = tau;
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;
    res("theta0") = theta0;
    res("theta1") = theta1;

    return res;
}


// [[Rcpp::export]]
List brute_force(mat data, int buff=10, float regularizer=1., float update_w=1.,
                float update_change=0.9, float tol=0.00001,
                int mapping_iter=1){
    /* estimates the single changepoint by brute force
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix of data
     *  buff : int
     *      buffer to keep proposals from edge
     *  regularizer : float
     *      see mapping
     *  update_w : float
     *      see mapping
     *  update_change : float
     *      see mapping
     *  tol : float
     *      see mapping
     *  mapping_iter : int
     *      see mapping
     *
     * Returns
     * -------
     *
     *  int estimate of tau
     */

    int N = data.n_rows;
    int P = data.n_cols;
    float ll;

    mat S_inv = inv(cov(data));

    mat S0;
    mat S1;
    mat theta0;
    mat theta1;

    float temp_regularizer;

    List ll_res;

    vec ll_l = zeros(N - 2 * buff);

    for (int i = buff; i <= N - buff; i++){

        theta0 = S_inv;
        theta1 = S_inv;

        S0 = cov(data.rows(0, i-1));
        S1 = cov(data.rows(i, N-1));

        temp_regularizer = regularizer * sqrt(log(P) / i);
        theta0 = mapping(S0, theta0, update_w, update_change,
                         temp_regularizer, mapping_iter, tol);
        temp_regularizer = regularizer * sqrt(log(P) / (N - i));
        theta1 = mapping(S1, theta1, update_w, update_change,
                         temp_regularizer, mapping_iter, tol);

        ll_res = log_likelihood(data, theta0, theta1, i, regularizer);
        ll = ll_res("ll_mod");


        ll_l(i - buff) = ll;
    }

    int min_tau = -1;
    // TODO fix this
    float ll_min = 1e20;
    for (int i = 0; i <= N - 2 * buff; i++){
        if (ll_l(i) < ll_min){
            min_tau = i + buff;
        }
    }

    theta0 = S_inv;
    theta1 = S_inv;

    S0 = cov(data.rows(0, min_tau-1));
    S1 = cov(data.rows(min_tau, N-1));

    temp_regularizer = regularizer * sqrt(log(P) / min_tau);
    theta0 = mapping(S0, theta0, update_w, update_change,
                     temp_regularizer, mapping_iter, tol);
    temp_regularizer = regularizer * sqrt(log(P) / (N - min_tau));
    theta1 = mapping(S1, theta1, update_w, update_change,
                     temp_regularizer, mapping_iter, tol);

    ll_res = log_likelihood(data, theta0, theta1,
                            min_tau, regularizer);
    float ll0 = ll_res("ll0");
    float ll1 = ll_res("ll1");
    float ll_mod = ll_res("ll_mod");

    List res;
    res("tau") = min_tau;
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;
    res("theta0") = theta0;
    res("theta1") = theta1;

    return res;
}


// [[Rcpp::export]]
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

    double val;
    double sign;

    float det0 = log_det(val, sign, theta0);
    float det1 = log_det(val, sign, theta1);

    float ll = tau * 0.5 * (-det0 + tr_TdS0);
    ll += regularizer * sqrt(log(P) / tau) * norm(theta0, 1);
    ll += (N - tau) * 0.5 * (-det1 + tr_TdS1);
    ll += regularizer * sqrt(log(P) / (N - tau)) * norm(theta1, 1);
    ll *= -1;

    vec ll_l = zeros(N - 2 * buff);

    ll_l(tau) = ll;

    mat Sp0 = S0;
    mat Sp1 = S1;

    mat TdSp0;
    mat TdSp1;

    float tr_TdSp0;
    float tr_TdSp1;

    for(int i = tau - 1; i >= buff; --i){

        mat op_data = zeros(P, 1);
        op_data.col(0) = data.col(i);
        mat rank_one_update = kron(op_data, trans(op_data));

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

        ll_l(i) = ll;

    }

    Sp0 = S0;
    Sp1 = S1;

    for(int i = tau + 1; i <= N - buff; i++){

        mat op_data = zeros(P, 1);
        op_data.col(0) = data.col(i);
        mat rank_one_update = kron(op_data, trans(op_data));

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

        ll_l(i) = ll;

    }

    int min_tau = -1;
    // TODO fix this
    float ll_min = 1e20;
    for (int i = 0; i <= N - 2 * buff; i++){
        if (ll_l(i) < ll_min){
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
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;

    return res;
}


// [[Rcpp::export]]
List rank_one(mat data, int buff=10, float regularizer=1., int tau=-1,
              int max_iter=25, float update_w=1., float update_change=0.9,
              int mapping_iter=1, float tol=0.00001){
    /* Does the rank one estimateion of the changepoint
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix containing the data for estimateion, note data
     *      should be mean centered
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
     *  int estimate of tau
     */

    int N = data.n_rows;
    int P = data.n_cols;

    if (tau == -1){
        tau = randi(1, distr_param(buff, N - buff))(0);
    }

    int iterations = 0;

    mat S_inv = inv(cov(data));

    mat theta0 = S_inv;
    mat theta1 = S_inv;

    mat S0;
    mat S1;
    float temp_regularizer;

    List res;
    List ll_res;

    float ll0;
    float ll1;
    float ll_mod;

    while (iterations < max_iter){

        S0 = cov(data.rows(0, tau-1));
        S1 = cov(data.rows(tau, N-1));

        temp_regularizer = regularizer * sqrt(log(P) / tau);
        theta0 = mapping(S0, theta0, update_w, update_change,
                         temp_regularizer, mapping_iter, tol);
        temp_regularizer = regularizer * sqrt(log(P) / (N - tau));
        theta1 = mapping(S1, theta1, update_w, update_change,
                         temp_regularizer, mapping_iter, tol);

        ll_res = log_likelihood_rank_one(data, S0, S1, theta0, theta1, buff,
                                         tau, regularizer);
        tau = ll_res("tau");
        ll0 = ll_res("ll0");
        ll1 = ll_res("ll1");
        ll_mod = ll_res("ll_mod");
        iterations += 1;
    }
    res("tau") = tau;
    res("ll0") = ll0;
    res("ll1") = ll1;
    res("ll_mod") = ll_mod;
    res("theta0") = theta0;
    res("theta1") = theta1;

    return res;
}


// [[Rcpp::export]]
vec binary_segmentation(mat data, float thresh, int method, int buff,
                        float regularizer){
    /* handles binary segmentation
     *
     * Parameters
     * ----------
     *
     *  data : mat
     *      N x P matrix of data
     *  thresh : float
     *      threshold for stopping
     *  method : method to be used
     *  0 - simulated annealing
     *  1 - rank one
     *  2 - brute force
     *  buff : int
     *      distance from edge
     *
     *  Returns
     *  -------
     *
     *  int estimate of tau
     */

    int N = data.n_rows;
    int P = data.n_cols;

    //float edge_check = buff / N;

    float temp_regularizer;

    // TODO this system is awful
    vec cp;
    vec temp_cp;
    vec ll_l;
    vec temp_ll_l;
    vec state;
    vec temp_state;
    mat datat;

    int Nt;
    int taut;
    int ll_counter;
    int cp_counter;
    int state_counter;
    List res;
    bool cond1;
    bool cond2;
    bool cond3;
    bool cond4;
    bool cond;

    float ll_mod;

    // handle the first change point
    if (method == 0){
        res = simulated_annealing(data);
    }
    if (method == 1){
        res = rank_one(data);
    }
    if (method == 2){
        res = brute_force(data);
    }

    cp = zeros(3);
    cp(1) = res("tau");
    cp(2) = N;

    ll_l = zeros(2);
    ll_l(0) = res("ll0");
    ll_l(1) = res("ll1");

    state = zeros(2);
    state(0) = 1;
    state(1) = 1;

    while (sum(state) > 0){

        // TODO a lot of this could probably be simplified
        temp_ll_l = zeros(ll_l.n_elem * 2);
        temp_cp = zeros(cp.n_elem * 2);
        temp_state = zeros(state.n_elem * 2);
        ll_counter = 0;
        cp_counter = 0;
        state_counter = 0;

        for(int i = 1; i < cp.n_elem; i++){

            if (state(i) == 1){
                datat = data.rows(cp(i-1), cp(i));
                Nt = datat.n_rows;
                temp_regularizer = regularizer * sqrt(log(P) / Nt);

                if (Nt > 2 * (buff + 1)){
                    // TODO fill these in/add error checking
                    if (method == 0){
                        res = simulated_annealing(datat);
                    }
                    else if (method == 1){
                        res = rank_one(datat);
                    }
                    else if (method == 2){
                        res = brute_force(datat);
                    }
                    taut = res("tau");
                    ll_mod = res("ll_mod");

                    cond1 = (ll_mod - ll_l(i - 1)) > thresh * P;
                    cond2 = ll_mod < 1e15;
                    cond3 = taut < Nt - buff;
                    cond4 = taut > buff;
                    cond = cond1 and cond2 and cond3 and cond4;
                }
                else {
                    cond = false;
                }

                if (cond) {
                    temp_ll_l(ll_counter) = res("ll0");
                    ll_counter += 1;
                    temp_ll_l(ll_counter) = res("ll1");
                    ll_counter += 1;

                    temp_cp(cp_counter) = taut + cp(i - 1);
                    cp_counter += 1;
                    temp_cp(cp_counter) = cp(i);
                    cp_counter += 1;

                    temp_state(state_counter) = 1;
                    state_counter += 1;
                    temp_state(state_counter) = 1;
                    state_counter += 1;
                }
                else {
                    temp_cp(cp_counter) = cp(i);
                    cp_counter += 1;
                    temp_state(state_counter) = 1;
                    state_counter += 1;
                }
            }
            else {
                temp_cp(cp_counter) = cp(i);
                cp_counter += 1;
                temp_state(state_counter) = 1;
                state_counter += 1;
            }
        }
        ll_l = temp_ll_l.head(ll_counter);
        cp = temp_cp(cp_counter);
        state = temp_state(state_counter);
    }

    return cp.subvec(1, cp.n_elem - 1);
}
