/**
 * \file iMatrixDiagPoisson_n.cpp
 * \author Felipe Acosta
 * \date 2015-04-13
 * \brief This function calculates the information matrix for the Poisson regression case with normal 
 * random effects for the diagonal case.
 * Arguments:
 * beta:      The fixed effects coefficients.
 * sigma:     Matrix with r rows. The covariance matrices for the random effects. There are 'r' 
 *            different covariance 
 *            matrices, one matrix per row. The first number of each row is the dimension of each 
 *            covariance matrix. The matrix is reconstructed with the function 'getSigma'.
 * sigmaType: Covariance matrix types, in case of a diagonal matrix the determinant and inverse have 
 *            closed forms. The types are:
 *            0 - diagonal
 * u:         Initial value for the vector of random effects.
 * kY:        Observations, 0 for failure and 1 for success.
 * kX:        Design matrix for fixed effects.
 * kZ:        Design matrix for random effects.
 */

#include "RcppArmadillo.h"
#include "mcemGLM.h"

using namespace Rcpp;
// [[Rcpp::depends("RcppArmadillo")]]

// [[Rcpp::export]]
arma::mat iMatrixDiagPoissonCpp_n(const arma::vec& beta, const arma::mat& sigma, const arma::mat& uSample, 
const arma::vec& kKi, const arma::vec& kY, const arma::mat& kX, const arma::mat& kZ, 
int B, double sd0) {
  int kP = kX.n_cols;  /** Dimension of Beta */
  int kR = kKi.n_elem; /** Number of random effects */
  
  //arma::mat uSample(B, kR); /** MCMC sample from U */
  //uSample = uSamplerPoissonCpp_n(beta, sigma, u, kY, kX, kZ, B, sd0);
  
  arma::vec g0(kP + kR); /** Gradient vector */
  g0.fill(0);
  arma::mat h0(kP + kR, kP + kR); /** Hessian Matrix */
  h0.fill(0);
  arma::mat iMatrix(kP + kR, kP + kR); /** Information Matrix */
  iMatrix.fill(0);
  
  for (int i = 0; i < B; i++) {
    g0 = loglikelihoodPoissonGradientCpp_n(beta, sigma, kKi, uSample.row(i).t(), kY, kX, kZ);
    h0 = loglikelihoodPoissonHessianCpp_n(beta, sigma, kKi, uSample.row(i).t(), kY, kX, kZ);
    iMatrix += (-h0 - g0 * g0.t()) / (double) B;
  }
  
  //iMatrix <-  iMatrix + (h0 - g0 %*% t(g0)) / ctrl$MCit
  
  return(iMatrix);
}
