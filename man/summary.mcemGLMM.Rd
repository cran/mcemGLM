\name{summary.mcemGLMM}
\alias{summary}
\alias{summary.mcemGLMM}
\title{Summary method for mcemGLMM objects}
\description{
  Summary for an object obtained from \code{mcemGLMM}.  
}
\usage{
\method{summary}{mcemGLMM}(object, ...)
}
\arguments{
  \item{object}{a model fitted with the mcemGLMM function.}
  \item{...}{additional arguments.}
}
\value{
  A list with the following items:
  \describe{
    \item{coefficients}{a list with the fixed effects coefficients 
      and the predicted random effects.}
    \item{var.est}{the estimated variances for each variance component.}
    \item{std.err}{the standard errors for the fixed effects 
      coefficients and the variance estimates.}
    \item{z.val}{z test values for the fixed effects coefficients and 
      the variance estimators.}
  }
}
\details{
The function prints a table for Wald tests for the fixed effect 
coefficients and the variance estimators. For the negative binomial 
and the gamma distributions the estimate of \eqn{\alpha} is reported 
with its respective standard error.
}
\author{
Felipe Acosta Archila <acosta@umn.edu>
}
\keyword{glmm}
