// [[Rcpp::depends(RcppArmadillo)]]

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <RcppArmadillo.h>

using namespace std;
using namespace Rcpp;
using namespace arma;


mat theta_sampler(mat theta, mat nd, vec ndsum, double alpha){
    /* updates the theta estimate
     *
     * Parameters
     * ----------
     *
     *  theta : mat
     *      N x K matrix of doc-topic proportions
     *  nd : mat
     *      N x K matrix of number of words assigned to each topic for
     *      each document
     *  ndsum : vec
     *      N length vec of sums for all words in doc
     *  alpha : scalar
     *      prior for theta
     *
     * Returns
     * -------
     *  update version of theta
     */

    int D = theta.n_rows;
    int K = theta.n_cols;

    for(int d = 0; d < D; d++){
        for(int k = 0; k < K; k++){
            theta(d,k) = (nd(d,k) + alpha) / (ndsum(d) + K * alpha);
        }
    }
    return(theta);
}


mat phi_sampler(mat phi, mat nw, vec nwsum, double beta){
    /* updates the phi estimate
     *
     * Parameters
     * ----------
     *
     *  phi : mat
     *      K x V matrix of topic-term proportions
     *  nw : mat
     *      K x V matrix of number of documents assigned to each topic for
     *      each word
     *  nwsum : vec
     *      K length vec of sums for all words assigned to topic
     *  beta : scalar
     *      prior for phi
     *
     * Returns
     * -------
     *  update version of phi
     */

    int K = phi.n_rows;
    int V = phi.n_cols;

    for(int k = 0; k < K; k++){
        for(int v = 0; v < V; v++){
            phi(k,v) = (nw(v,k) + beta) / (nwsum(k) + V * beta);
        }
    }
    return(phi);
}


List z_sampler(int d, int v, mat corpus, mat z, mat nd, mat nw, vec ndsum,
               vec nwsum, double alpha, double beta){
    /* samples the topic for each document/word
     *
     * Parameters
     * ----------
     *
     *  d : scalar
     *      index of current doc
     *  v : scalar
     *      index of current word
     *  corpus : mat
     *      N x V original DTM
     *  z : mat
     *      N x V current topic assignments for each word
     *  nd : mat
     *      N x K matrix of number of words assigned to each topic for
     *      each document
     *  nw : mat
     *      K x V matrix of number of documents assigned to each topic for
     *      each word
     *  ndsum : vec
     *      N length vec of sums for all words in doc
     *  nwsum : vec
     *      K length vec of sums for all words assigned to topic
     *  alpha : scalar
     *      prior for theta
     *  beta : scalar
     *      prior for phi
     *
     * Returns
     * -------
     *  List of updated elements
     *      topic
     *      nw
     *      nd
     *      nwsum
     *      ndsum
     */

    int D = corpus.n_rows;
    int K = nd.n_cols;
    int V = corpus.n_cols;
    vec p = zeros(K);

    int topic = z(d,v);
    int vcount = corpus(d,v);
    nw(v,topic) -= vcount;
    nd(d,topic) -= vcount;
    nwsum(topic) -= vcount;
    ndsum(d) -= vcount;

    double Vbeta = V * beta;
    double Kalpha = K * alpha;

    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++){
        p(k) = (nw(v,k) + beta) / (nwsum(k) + Vbeta) *
                (nd(d,k) + alpha) / (ndsum(d) + Kalpha);
    }
    // cumulative multinomial parameters
    for (int k = 1; k < K; k++){
        p(k) += p(k - 1);
    }

    // scaled sample because of unnormalized p()
    double u = R::runif(0,1) * p(K - 1);
//    double u = ((double)rand() / RAND_MAX) * p(K - 1);

    for(int topic = 0; topic < K; topic++) {
        if (p(topic) > u) {
            break;
        }
    }

    nw(v,topic) += vcount;
    nd(d,topic) += vcount;
    nwsum(topic) += vcount;
    ndsum(d) += vcount;

    List res;
    res("topic") = topic;
    res("nw") = nw;
    res("nd") = nd;
    res("nwsum") = nwsum;
    res("ndsum") = ndsum;
    return(res);
}

// LDA black box model
//' @name latent_dirichlet_allocation
//'
//' @title Estimates LDA topic model.
//'
//' @description Estimates a LDA topic model using collapsed Gibbs
//'              sampling.
//'
//' @param corpus Matrix corresponding to the DTM.
//' @param latent_vars List containing the set of variables to be
//'         updated as part of the estimation procedure.
//' @param niters Number of iterations for LDA to run.
//' @param alpha Prior for theta.
//' @param beta Prior for phi.
//'
//' @return latent_vars Updated version of List that was taken
//'         as input.
//'
//' @author \packageMaintainer{changepointsHD}
// [[Rcpp::export]]
List latent_dirichlet_allocation(arma::mat corpus, List latent_vars,
                                 int niters=1500, double alpha=1,
                                 double beta=1){
    /* handles the LDA estimation
     *
     * Parameters
     * ----------
     *  corpus : mat
     *      N x V DTM
     *  latent_vars : List
     *      container for all the latent variables worked with during
     *      estimation:
     *          z : N x V topic assignments
     *          theta : N x K doc-top props
     *          phi : K x V top-term props
     *          nw : V x K doc counts assigned topic k
     *          nd : D x K word counts assigned topic k
     *          nwsum : K total counts assigned topic k
     *          ndsum : D total words doc d
     *  niter : scalar
     *      number of LDA iterations
     *  alpha : scalar
     *      prior for theta
     *  beta : scalar
     *      prior for phi
     *
     *  Returns
     *  -------
     *  List, see latent_vars
     */

    mat z = latent_vars("z");
    mat theta = latent_vars("theta");
    mat phi = latent_vars("phi");
    mat nw = latent_vars("nw");
    mat nd = latent_vars("nd");
    vec nwsum = latent_vars("nwsum");
    vec ndsum = latent_vars("ndsum");

    int D = corpus.n_rows;
    int V = corpus.n_cols;

    int topic;
    List z_update;

    for (int liter = 1; liter <= niters; liter++){
        // for all z_i
        for (int d = 0; d < D; d++) {
            for (int v = 0; v < V; v++) {
                z_update = z_sampler(d, v, corpus, z, nd, nw, ndsum, nwsum,
                                     alpha, beta);
                topic = z_update("topic");
                mat nw = z_update("nw");
                mat nd = z_update("nd");
                vec nwsum = z_update("nwsum");
                vec ndsum = z_update("ndsum");
                z(d,v) = topic;
            }
        }
        printf("LDA Iterations: %d\n", liter);
    }
    theta = theta_sampler(theta, nd, ndsum, alpha);
    phi = phi_sampler(phi, nw, nwsum, beta);

    latent_vars("z") = z;
    latent_vars("theta") = theta;
    latent_vars("phi") = phi;
    latent_vars("nw") = nw;
    latent_vars("nd") = nd;
    latent_vars("nwsum") = nwsum;
    latent_vars("ndsum") = ndsum;
    return(latent_vars);
}


// LDA log-likelihood
//' @name latent_dirichlet_allocation_ll
//'
//' @title Estimates log-likelihood for LDA.
//'
//' @description Generates the log-likelihood for a corresponding set of
//'              latent variables.
//'
//' @param corpus Matrix corresponding to the DTM.
//' @param latent_vars List containing the set of variables to be
//'         updated as part of the estimation procedure.
//'
//' @return Log-likelihood estimate.
//'
//' @author \packageMaintainer{changepointsHD}
// [[Rcpp::export]]
double latent_dirichlet_allocation_ll(arma::mat corpus, List latent_vars){
    /* generates the log likelihood for the corresponding LDA estimates
     *
     * Parameters
     * ----------
     *
     *  corpus : mat
     *      N x V DTM
     *  latent_vars : List
     *      container for all the latent variables worked with during
     *      estimation:
     *          z : N x V topic assignments
     *          theta : N x K doc-top props
     *          phi : K x V top-term props
     *          nw : V x K doc counts assigned topic k
     *          nd : D x K word counts assigned topic k
     *          nwsum : K total counts assigned topic k
     *          ndsum : D total words doc d
     *
     * Returns
     * -------
     *  double ll estimate
     */

    int D = corpus.n_rows;
    int V = corpus.n_cols;
    mat phi = latent_vars("phi");
    mat theta = latent_vars("theta");
    mat z = latent_vars("z");

    double ll = 0.0;
    for (int d = 0; d<D; d++){
        for(int v = 0; v<V; v++){
            ll += corpus(d,v) * log(phi(z(d,v),v) * theta(d,z(d,v)));
        }
    }
    return(ll);
}
