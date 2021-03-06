/** 
 * \file uSamplerNegBinom_t.cpp
 * \author Felipe Acosta
 * \date 2015-04-27
 * \brief This function performs an MCMC run on the random effects. The arguments are the same arguments used in
 * loglikelihood with an extra argument 'B' which indicates the MCMC sample size and the argument 'u' now 
 * indicates the intial value for the chain.
 */


#include "RcppArmadillo.h"
#include "mcemGLM.h"

using namespace Rcpp;
// [[Rcpp::depends("RcppArmadillo")]]

double logAcceptNegBinom_t(const arma::vec& beta, const arma::mat& sigma, double alpha, const arma::vec& sigmaType, const arma::vec& ucurrent, 
const arma::vec& uproposed, const arma::vec& df, const arma::vec& kKi, const arma::vec& kLh, const arma::vec& kLhi, 
const arma::vec& kY, const arma::mat& kX, const arma::mat& kZ) {
  return min0(0.0, loglikelihoodNegBinomCpp_t(beta, sigma, alpha, sigmaType, uproposed, df, kKi, kLh, kLhi, kY, kX, kZ) 
  - loglikelihoodNegBinomCpp_t(beta, sigma, alpha, sigmaType, ucurrent, df, kKi, kLh, kLhi, kY, kX, kZ));
}

// [[Rcpp::export]]
arma::mat uSamplerNegBinomCpp_t(const arma::vec& beta, const arma::mat& sigma, double alpha, const arma::vec& sigmaType, const arma::vec& u, 
const arma::vec& df, const arma::vec& kKi, const arma::vec& kLh, const arma::vec& kLhi, const arma::vec& kY, 
const arma::mat& kX, const arma::mat& kZ, int B, double sd0) {
  RNGScope scope;
  int kK = u.n_rows;
  
  arma::mat usample(B, kK);
  arma::vec ucurrent(kK);
  arma::vec uproposed(kK);
  ucurrent = u;
  usample.row(0) = ucurrent.t();
  
  for (int i = 1; i < B; i++){
    // uproposed = rnorm(kK, 0, sd0);
    for (int j = 0; j < kK; j++) {
      uproposed(j) = rnorm(1, 0 , sd0 * sqrt(sigma(j, j)))(0);
    }
    uproposed += ucurrent;
    if (log(R::runif(0, 1)) < logAcceptNegBinom_t(beta, sigma, alpha, sigmaType, ucurrent, uproposed, df, kKi, kLh, kLhi, kY, kX, kZ)) {
      ucurrent = uproposed;
    }
    usample.row(i) = ucurrent.t();
  }
  
  return usample;
}
