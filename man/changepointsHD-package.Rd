\name{changepointsHD-package}
\alias{changepointsHD-package}
\alias{changepointsHD}
\docType{package}
\title{
\packageTitle{changepointsHD}
}
\description{
\packageDescription{changepointsHD}
}
%\details{
%\tabular{ll}{
%Package: \tab \packagePackage{changepoints} \cr
%Type: \tab \packageType{changepoints} \cr
%Version: \tab \packageVersion{changepoints} \cr
%Date: \tab \packageDate{changepoints} \cr
%Licence: \tab packageLicense{changepoints} \cr
%}
\author{
\packageAuthor{changepointsHD}

Maintainer: \packageMaintainer{changepointsHD}
}
%\references{
%~~ Literature or other references for background information ~~
%}
%~~ Optionally other standard keywords, one per line, from file KEYWORDS ~~
%~~ in the R documentation directory ~~
\keyword{ package }
%\seealso{
%~~ Optional links to other man pages, e.g. ~~
%~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
%}
%\examples{
%scp_data = read.table("scp.txt")
%scp_data = as.matrix(scp_data)

%# prox gradient black-box method
%cov_est = cov(scp_data)
%init = solve(cov_est)
%res_map = prox_gradient_mapping(scp_data, init, 0.1, 0.99, 0.1, 100, 1e-20)

%# prox gradient black-box ll
%res_ll = prox_gradient_ll(scp_data, res_map, 0.1)

%prox_gradient_params=list()
%prox_gradient_params$update_w = 0.1
%prox_gradient_params$update_change = 0.99
%prox_gradient_params$regularizer = 0.1
%prox_gradient_params$max_iter = 1
%prox_gradient_params$tol = 1e-5

%prox_gradient_ll_params=list()
%prox_gradient_ll_params$regularizer = 0.1

%simulated_annealing_params=list()
%simulated_annealing_params$buff=10

%res_sa = simulated_annealing(scp_data, init, prox_gradient_mapping,
%                             prox_gradient_ll, buff=10,
%                             bbmod_method_params=prox_gradient_params,
%                             bbmod_ll_params=prox_gradient_ll_params)
%res_bf = brute_force(scp_data, init, prox_gradient_mapping,
%                     prox_gradient_ll, buff=10,
%                     bbmod_method_params=prox_gradient_params,
%                     bbmod_ll_params=prox_gradient_ll_params)
%res_ro = rank_one(scp_data, init, update_w=0.1, regularizer=0.1)

%mcp_data = read.table("mcp.txt")
%mcp_data = as.matrix(mcp_data)

%res_bs = binary_segmentation(mcp_data, init, simulated_annealing,
%                             prox_gradient_mapping,
%                             prox_gradient_ll, buff=10,
%                             cp_method_params=simulated_annealing_params,
%                             bbmod_method_params=prox_gradient_params,
%                             bbmod_ll_params=prox_gradient_ll_params)
%}
