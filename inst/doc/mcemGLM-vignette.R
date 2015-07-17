### R code from vignette source 'mcemGLM-vignette.Rnw'

###################################################
### code chunk number 1: mcemGLM-vignette.Rnw:129-131
###################################################
require(mcemGLM)
data("simData.rdata")


###################################################
### code chunk number 2: mcemGLM-vignette.Rnw:133-136
###################################################
set.seed(23786)
simData$count <- simData$count+rpois(200, 3)
simData$count2 <- simData$count*(1+rpois(200, 2))


###################################################
### code chunk number 3: mcemGLM-vignette.Rnw:138-140
###################################################
head(simData)
summary(simData)


###################################################
### code chunk number 4: mcemGLM-vignette.Rnw:151-156
###################################################
fit0 <- mcemGLMM(fixed = obs ~ x1 + z1, 
                random = ~0+z2, 
                  data = simData, 
                family = "bernoulli", 
                vcDist = "normal")


###################################################
### code chunk number 5: mcemGLM-vignette.Rnw:166-167
###################################################
summary(fit0)


###################################################
### code chunk number 6: mcemGLM-vignette.Rnw:172-173
###################################################
anova(fit0)


###################################################
### code chunk number 7: mcemGLM-vignette.Rnw:178-188
###################################################
ctr0 <- rbind("D1 - D2" = c(0, 0,-1, 0, 0, 0),
              "D1 - D3" = c(0, 0, 0,-1, 0, 0),
              "D1 - D4" = c(0, 0, 0, 0,-1, 0),
              "D1 - D5" = c(0, 0, 0, 0, 0,-1),
              "D2 - D3" = c(0, 0, 1,-1, 0, 0),
              "D2 - D4" = c(0, 0, 1, 0,-1, 0),
              "D2 - D5" = c(0, 0, 1, 0, 0,-1),
              "D3 - D4" = c(0, 0, 0, 1,-1, 0),
              "D3 - D5" = c(0, 0, 0, 1, 0,-1),
              "D4 - D5" = c(0, 0, 0, 0, 1,-1))


###################################################
### code chunk number 8: mcemGLM-vignette.Rnw:191-192
###################################################
contrasts.mcemGLMM(object = fit0, ctr.mat = ctr0)


###################################################
### code chunk number 9: mcemGLM-vignette.Rnw:199-200
###################################################
plot(simData$x1, predict(fit0, type = "response"), col = simData$z1, xlab = "x1")


###################################################
### code chunk number 10: mcemGLM-vignette.Rnw:209-210
###################################################
plot(simData$x1, residuals(fit0, type = "deviance"))


###################################################
### code chunk number 11: mcemGLM-vignette.Rnw:218-219
###################################################
plot(simData$x1, residuals(fit0, type = "pearson"))


###################################################
### code chunk number 12: mcemGLM-vignette.Rnw:229-234
###################################################
par(mfrow = c(1, 2))
ts.plot(fit0$mcemEST, main = "MLEs estimates", 
        xlab = "EM Iteration", ylab = "MLE value")
ts.plot(fit0$loglikeVal, main = "Loglikelihood values", 
        xlab = "EM iteration", ylab = "Likelihood")


###################################################
### code chunk number 13: mcemGLM-vignette.Rnw:244-245
###################################################
ts.plot(fit0$randeff[, 1], xlab = "MCMC iteration")


###################################################
### code chunk number 14: mcemGLM-vignette.Rnw:251-252
###################################################
colMeans(fit0$randeff)


###################################################
### code chunk number 15: mcemGLM-vignette.Rnw:259-260
###################################################
ts.plot(fit0$loglikeMCMC, xlab = "MCMC iteration")


###################################################
### code chunk number 16: mcemGLM-vignette.Rnw:270-276
###################################################
fit1 <- mcemGLMM(fixed = obs ~ x1 + x2 + x3, 
                random = list(~0+z1, ~0+z1:z2), 
                  data = simData, 
                family = "bernoulli", 
                vcDist = "t", 
                df = c(5, 5))


###################################################
### code chunk number 17: mcemGLM-vignette.Rnw:281-283
###################################################
summary(fit1)
anova(fit1)


###################################################
### code chunk number 18: mcemGLM-vignette.Rnw:287-291
###################################################
ctr1 <- rbind(   "blue - red" = c(0, 0, 0,-1, 0),
              "blue - yellow" = c(0, 0, 0, 0,-1),
               "red - yellow" = c(0, 0, 0, 1,-1))
contrasts.mcemGLMM(object = fit1, ctr.mat = ctr1)


###################################################
### code chunk number 19: mcemGLM-vignette.Rnw:295-302
###################################################
fit2 <- mcemGLMM(fixed = obs ~ x1 + x2, 
                random = list(~0+z1, ~0+z1:z2), 
                  data = simData, 
                family = "bernoulli", 
                vcDist = "t", 
                    df = c(5, 5),
                controlEM = list(EMit = 3))


###################################################
### code chunk number 20: mcemGLM-vignette.Rnw:305-306
###################################################
anova(fit1, fit2)


###################################################
### code chunk number 21: mcemGLM-vignette.Rnw:311-316
###################################################
fit3 <- mcemGLMM(fixed = count ~ x1 + x2 + x3, 
                random = list(~0+z2), 
                  data = simData, 
                family = "poisson", 
                vcDist = "normal")


###################################################
### code chunk number 22: mcemGLM-vignette.Rnw:319-322
###################################################
summary(fit3)
anova(fit3)
contrasts.mcemGLMM(object = fit3, ctr.mat = ctr1)


###################################################
### code chunk number 23: mcemGLM-vignette.Rnw:328-330
###################################################
plot(simData$x1, predict(fit3), 
     main = "Predicted response values", xlab = "x1")


###################################################
### code chunk number 24: mcemGLM-vignette.Rnw:338-343
###################################################
par(mfrow = c(1, 2))
ts.plot(fit3$mcemEST, main = "MLEs estimates", 
        xlab = "EM Iteration", ylab = "MLE value")
ts.plot(fit3$loglikeVal, main = "Loglikelihood values", 
        xlab = "EM iteration", ylab = "Likelihood")


###################################################
### code chunk number 25: mcemGLM-vignette.Rnw:351-359
###################################################
fit4 <- mcemGLMM(fixed = count2 ~ x1 + x2 + x3, 
                random = list(~0+z1, ~0+z1:z2), 
                  data = simData, 
                family = "negbinom", 
                vcDist = "normal")
summary(fit4)
anova(fit4)
contrasts.mcemGLMM(object = fit4, ctr.mat = ctr1)


###################################################
### code chunk number 26: mcemGLM-vignette.Rnw:364-365
###################################################
plot(simData$x1, predict(fit4))


###################################################
### code chunk number 27: mcemGLM-vignette.Rnw:372-377
###################################################
par(mfrow = c(1, 2))
ts.plot(fit4$mcemEST, main = "MLEs estimates", 
        xlab = "EM Iteration", ylab = "MLE value")
ts.plot(fit4$loglikeVal, main = "Likelihood value", 
        xlab = "EM Iteration", ylab = "Likelihood")


