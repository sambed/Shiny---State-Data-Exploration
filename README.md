---
output: word_document
---
```{r, echo=FALSE}
setwd('C:/Users/adhikas/Documents')
load(JAN27.RDATA)
library(caret)

##Get the 25-NN fit

fit25<- knnreg(as.matrix(x),y,k=25)
plot(x,y)

##Plot the 25-NN fitted lie

curve(predict(fit25,x),add=TRUE)
## THE END POINTS have all same 25 neighbors so same results
## the middle points goes up and down based on data

##Try  10 points or get the 10-nnn fit

fit10<- knnreg(as.matrix(x),y,k=10)
curve(predict(fit10,x),add=TRUE, col = 'blue')

## is one fit beter than another??
## y = reducible + irreducible
##

fit2<- knnreg(as.matrix(x),y,k=2)
curve(predict(fit2,x),add=TRUE, col = 'red')  ##fits data too well

##as k decreases you're making the model complicated and less flexible.
## new prediction may suck!

##lot if simulated data from y = f(x) +e
##based on simulated data based on 5 neihborest from 5,10,15 NN 
## and i wanna compare those models to the truth. 


##PLOT the 10 nearest nn models that would restuls from
## all 1000 simulated datasets in sim.y

plot(x,y,type='n')
for (i in 1:1000){
  fit <-knnreg(as.matrix(x),sim.y[,i],k=10)
  curve(predict(fit,x),add=T)
  
  
}

curve(f(x),add=T,col='blue',lwd=3)


##PLOT the 25 nearest nn models that would restuls from
## all 1000 simulated datasets in sim.y

plot(x,y,type='n')
for (i in 1:1000){
  fit <-knnreg(as.matrix(x),sim.y[,i],k=25)
  curve(predict(fit,x),add=T)
  
  
}

curve(f(x),add=T,col='blue',lwd=3)


##BANDS are tighter in this 25 case!

##PLOT the 5 nearest nn models that would restuls from
## all 1000 simulated datasets in sim.y

plot(x,y,type='n')
for (i in 1:1000){
  fit <-knnreg(as.matrix(x),sim.y[,i],k=5)
  curve(predict(fit,x),add=T)
  
  
}

curve(f(x),add=T,col='blue',lwd=3)

##more variablity even though we took some bias rom 10nn



###DRAFT data

###################
## STA 4/567     ##
## In-class code ##
## 01 29 2016    ##
###################

### Note: Some history of the draft lottery is given at:
#   https://www.sss.gov/About/History-And-Records/lotter1

## Load needed libraries
library(caret)  

## 1969 Draft lottery data
dat <- read.table("http://www.amstat.org/publications/jse/datasets/draft70yr.dat.txt")
colnames(dat) <- c("day","draftnum","month")

## Plot the data
with(dat,plot(day,draftnum,bty="l",xlab="Birthday (Day of the Year)",
              ylab="Draft Number"))

## Construct a simple linear regression model for predicting draft number from birthday
mod <- lm(draftnum~day,data=dat)
summary(mod)
abline(mod)

## Draft number appears to be decreasing in birthday.  Let's look 
## at month-by-month summaries
monthmeans <- with(dat,by(draftnum,month,mean))
monthmeans

## Add the month-by-month summaries to the plot
# Show lines separating months
brks <- c(31.5,60.5,91.5,121.5,152.5,182.5,213.5,244.5,274.5,305.5,335.5)
abline(v=brks,lty=3)

# Construct a new variable providing the month mean for each date
mm <- monthmeans[dat$month]

# Show the month means on the graph
lines(dat$day,mm,col="red")

#########################################################################
#########################################################################

## Fit kNN regression models for various values of k
mod5 <- with(dat,knnreg(as.matrix(day),draftnum,k=5))
mod10 <- with(dat,knnreg(as.matrix(day),draftnum,k=10))
mod20 <- with(dat,knnreg(as.matrix(day),draftnum,k=20))
mod30 <- with(dat,knnreg(as.matrix(day),draftnum,k=30))
mod40 <- with(dat,knnreg(as.matrix(day),draftnum,k=40))
mod50 <- with(dat,knnreg(as.matrix(day),draftnum,k=50))
mod100 <- with(dat,knnreg(as.matrix(day),draftnum,k=100))
mod150 <- with(dat,knnreg(as.matrix(day),draftnum,k=150))
mod200 <- with(dat,knnreg(as.matrix(day),draftnum,k=200))
mod250 <- with(dat,knnreg(as.matrix(day),draftnum,k=250))
mod300 <- with(dat,knnreg(as.matrix(day),draftnum,k=300))
mod350 <- with(dat,knnreg(as.matrix(day),draftnum,k=350))

## Construct a function to plot a KNN fit
plotres <- function(md,...){
  ## "md" is the name of the knnreg model
  with(dat,plot(day,draftnum,bty="l",xlab="Birthday (Day of the Year)",
                ylab="Draft Number",...))
  brks <- c(31.5,60.5,91.5,121.5,152.5,182.5,213.5,244.5,274.5,305.5,335.5)
  abline(v=brks,lty=3)
  lines(1:366,predict(md,1:366))
}

## Plot the KNN models
plotres(mod5,main="KNN Fit with k = 5")
plotres(mod10,main="KNN Fit with k = 10")
plotres(mod20,main="KNN Fit with k = 20")
plotres(mod30,main="KNN Fit with k = 30")
plotres(mod40,main="KNN Fit with k = 40")
plotres(mod50,main="KNN Fit with k = 50")
plotres(mod100,main="KNN Fit with k = 100")
plotres(mod150,main="KNN Fit with k = 150")
plotres(mod200,main="KNN Fit with k = 200")
plotres(mod250,main="KNN Fit with k = 250")
plotres(mod300,main="KNN Fit with k = 300")
plotres(mod350,main="KNN Fit with k = 350")

### Which model is the "best" in terms of prediction of new observations?
### (In other words, suppose the same procedure would be used for the draft in
###  later years. Which model would we expect to do the best job?)

###############################################################################
###############################################################################

## Option 1: Split data into training, testing set
set.seed(106)
intrain <- rbinom(366,1,.6)
train <- subset(dat,intrain==1)
test <- subset(dat,intrain==0)

## Check the fit of various kNN models, pick the "best"

# Here's the procedure on one model; we'll automate it soon
mod5t <- with(train,knnreg(as.matrix(day),draftnum,k=5))
pred5 <- with(test,predict(mod5t,as.matrix(day)))
testmse5 <- with(test,mean((draftnum-pred5)^2))
testrmse5 <- sqrt(testmse5)

# Let's look at this automatically for a bunch of different values of k
testrmses <- numeric(199)
trainrmses <- numeric(199)

for(nn in 2:200){
  trainmod <- with(train,knnreg(as.matrix(day),draftnum,k=nn))
  
  # Determine fit of model within training set
  trainpred <- with(train,predict(trainmod,as.matrix(day)))
  trainmse <- with(train,mean((draftnum-trainpred)^2))
  trainrmses[nn-1] <- sqrt(trainmse)
  
  # Determine performance of model on test set
  testpred <- with(test,predict(trainmod,as.matrix(day)))
  testmse <- with(test,mean((draftnum-testpred)^2))
  testrmses[nn-1] <- sqrt(testmse)
}

plot(2:200,trainrmses,type="l",ylim=c(70,130))
lines(2:200,testrmses,col="red")
text(20,95,"train")
text(20,110,"test",col="red")

which(testrmses==min(testrmses))


## Final model selected is a k=35+1=36 NN model
mod36 <- with(dat,knnreg(as.matrix(day),draftnum,k=36))
plotres(mod36,main="KNN Fit with k = 36")





###Otion 2 : Cross Validation
# let's see how cv works on a single model, in this case we;kk choose 5nn

## we'll perform 10 old CV , so we need to split data into 10 toughly equal subsets

set.seed(201)
subsets <- sample(rep_len(1:10,366))

##What do you wanna keep track of? sum of error 
cvsets <- numeric(10)

for (i in 1:10){
  train <-subset(dat,subsets != i)
  test <- subset(dat,subsets ==i)
  
  ##fit the moel in the temporray training set
  
  fit <- with(train,knnreg(as.matrix(day),draftnum, k =5))
  
  #predict values o the temporary test set and evaluatefit
  
  testpred <- with(test,predict(fit,as.matrix(day)))
  cvsses[i] <- with(test,sum((draftnum-testpred)^2))
}

cvsse <- sum(cvsses)
cvmse <- cvsse/366
cvrmse <- sqrt(cvmse)


###Now lets perform cv on eac of the KNN models from 2 to 200
## and use our results  to select the 'best' model complexity for ## predicting the observations

cvrmses <- numeric(199)

for (nn in 2:200){
  cvsses <- numeric(10)
  
  for (i in 1:10){
    train <-subset(dat,subsets != i)
    test <- subset(dat,subsets ==i)
    
    ##fit the moel in the temporray training set
    
    fit <- with(train,knnreg(as.matrix(day),draftnum, k =nn))
    
    #predict values o the temporary test set and evaluatefit
    
    testpred <- with(test,predict(fit,as.matrix(day)))
    cvsses[i] <- with(test,sum((draftnum-testpred)^2))
  }
  
  
}

cvrmses[nn-1] <- sqrt(sum(cvsses)/366)

which(cvrmses = min(cvrmses))
abline(v = 90, lty =3)


cvrmses[89]  ## 103.5


##02/03/2016










###Now lets perform cv on eac of the KNN models from 2 to 200
## and use our results  to select the 'best' model complexity for ## predicting the observations
###let;s akter iyr cide skigtk that we estimate SE rather than RMSE (OF PREDICTION). aND THEN WE;LL INCLUDE
## ESTIMATES OF THE 'STANDARD ERROR' OF OUR mse ESTIMAGES


cvmses <- numeric(199)
cvmses.se <-numeric(199)

##obtain the cvmse for each number of NN

for (nn in 2:500){
  subset.mses<- numeric(10)
  
  for (i in 1:10){
    train <-subset(dat,subsets != i)
    test <- subset(dat,subsets ==i)
    
    ##fit the moel in the temporray training set
    
    fit <- with(train,knnreg(as.matrix(day),draftnum, k =nn))
    
    #predict values o the temporary test set and evaluatefit
    
    testpred <- with(test,predict(fit,as.matrix(day)))
    subset.mses[i] <- with(test,mean((draftnum-testpred)^2))
  }
  cvmses[nn-1] <- mean(subset.mses)
  cvmses.se[nn-1] <-sd(subset.mses)/sqrt(10)
  
}

plot(2:500,cvmses,type='l')  ##plot cv error against ##nn

# identify the 0-se model

index.0se <- which(cvmses == min(cvmses))
index.0se
abline(v = index.0se+1, lty =3)

##identify the 1-se model
threshold <-cvmses[index.0se]+cvmses.se[index.0se]
abline(h=threshold,lty=2)
index1.se <-max(which(cvmses<=threshold))
index1.se

##brining down complexity isn't helping


plot(x,y)
dt <-data.frame(x=x, y =sim.y[,586
                              ])
with(dt,plot(x,y))


###HW1
load("M:/StatsLearning/Jan27.RData")
set.seed(889)
subsets <- sample(rep_len(1:10,101))
dt <-data.frame(x=x, y =sim.y[,586])

cvmses <- numeric(89)
cvmses.se <-numeric(89)

##obtain the cvmse for each number of NN

for (nn in 2:90){
  subset.mses<- numeric(10)
  
  for (i in 1:10){
    train <-subset(dt,subsets != i)
    test <- subset(dt,subsets ==i)
    
    ##fit the moel in the temporray training set
    
    fit <- with(train,knnreg(as.matrix(x),y, k =nn))
    
    #predict values o the temporary test set and evaluatefit
    
    testpred <- with(test,predict(fit,as.matrix(x)))
    subset.mses[i] <- with(test,mean((y-testpred)^2))
  }
  cvmses[nn-1] <- mean(subset.mses)
  cvmses.se[nn-1] <-sd(subset.mses)/sqrt(10)
  
}

plot(2:90,cvmses,type='l')  ##plot cv error against ##nn

# identify the 0-se model

index.0se <- which(cvmses == min(cvmses))
index.0se
abline(v = index.0se+1, lty =3)

##identify the 1-se model
threshold <-cvmses[index.0se]+cvmses.se[index.0se]
abline(h=threshold,lty=2)
index1.se <-max(which(cvmses<=threshold))
index1.se








```

