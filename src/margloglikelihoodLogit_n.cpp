/**
 * \file margloglikelihoodLogit_n.cpp
 * \author Felipe Acosta
 * \date 2015-08-03
 * \brief Despite its name, this function evaluates the Q function for the logistic regression case with normal 
 * random effects.
 * Arguments:
 * beta:      The fixed effects coefficients.
 * sigma:     Matrix with r rows. The covariance matrices for the random effects. There are 'r' 
 *            different covariance 
 *            matrices, one matrix per row. The first number of each row is the dimension of each 
 *            covariance matrix. The matrix is reconstructed with the function 'getSigma'.
 * sigmaType: Covariance matrix types, in case of a diagonal matrix the determinant and inverse have 
 *            closed forms. The types are:
 *            0 - diagonal
 * u:         Vector of random effects.
 * kY:        Observations, 0 for failure and 1 for success.
 * kX:        Design matrix for fixed effects.
 * kZ:        Design matrix for random effects.
 */


#include "RcppArmadillo.h"
#include "mcemGLM.h"

using namespace Rcpp;
// [[Rcpp::depends("RcppArmadillo")]]

double partialLogitCpp_n(const arma::vec& beta, const arma::mat& sigma, const arma::vec& u, const arma::vec& kY, const arma::mat& kX, const arma::mat& kZ) {
  double value = 0; /** The value to be returned */
  
  int nObs = kY.n_elem;
  int kP = kX.n_cols;  /** Dimension of Beta */
  int kK = kZ.n_cols;  /** Dimension of U */
  
  /** sum of yij * (wij - log(1 + ...))
   *  This corresponds to the 
  */
  for (int i = 0; i < nObs; i++) {
    double wij = 0;
    for (int j = 0; j < kP; j++) {
      wij += kX(i, j) * beta(j);
    }
    
    for (int j = 0; j < kK; j++) {
      wij += kZ(i, j) * u(j);
    }
    value += kY(i) * wij - log(1 + exp(wij));
  }
  
  return exp(value);
}

// [[Rcpp::export]]
double margloglikelihoodLogitCpp_n(const arma::vec& beta, const arma::mat& sigma, const arma::mat& u, const arma::vec& kY, const arma::mat& kX, const arma::mat& kZ) {
  int kM = u.n_rows;
  double loglike = 0;
  
  for (int i = 0; i < kM; i++) {
    loglike += partialLogitCpp_n(beta, sigma, u.row(i).t(), kY, kX, kZ);
  }
  
  return log(loglike / kM);
}

