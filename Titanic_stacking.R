##############################################
#stacking method from kaggle: stacked generalization
#k-fold stacking:
# Split the train set into k parts: train_1 to train_k
# do the usual cv fit: Fit a model on all training data excluding train_a, get prediction on train_a. 
# then we will have cv predictions on all training data. 
# repeat for other first-stage models. Suppose we have L such models.
# we would have L vectors of predictions of length n.
# use this n*L matrix as design matrix and true response of training data as "Y" to get another second-stage model.
# Finally, fit each of L models again on full training data. They are called L first-stage models.
# the second-stage model and L first-stage models will be used on test data to get the final prediction:
# 
library(randomForest)
library(dplyr)
library(ggplot2)
library(caret)
tr=read.csv("train.csv")
te=read.csv("test.csv")
full=bind_rows(tr,te)
# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)# .代表一个字符，.* 代表全部字符
# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')
# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare'
# Finally, grab surname from passenger name
full$Surname=gsub(",.*","",full$Name)

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')
full$Embarked[full$Embarked==""]="C"
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)
set.seed(1234)

#facterize all factors
full[,c(2,3,5,9,12,13,14,15,16)] <- lapply(full[,c(2,3,5,9,12,13,14,15,16)] , factor)
#impute age using random forest
imputefit=randomForest(Age~.,data=full[,-c(1,2,4,9,11,14,16)][! is.na(full$Age),],na.action = na.omit,xtest=full[,-c(1,2,4,6,9,11,14,16)][is.na(full$Age),],ntree=5000)
full$Age[is.na(full$Age)]=imputefit$test$predicted
#add a variable "lowclassmale", as it was reported low class male was left over.
full$singlemale=as.factor(full$Sex=="male" & full$Pclass==3)
#add  "womenchild", as women and children first rule.
full$womenchild=as.factor(full$Sex == "female" | full$Age < 15)

#change factors levels name
levels(full$Survived)=c("no","yes")

#split into train and test
tr1=full[1:891,]
te1=full[892:nrow(full),]
#keep variables of useful
usefulvar=c("Survived"  ,  "Pclass"    ,   "Sex"    ,     "Age"      ,   "SibSp"   ,    "Parch"   ,
            "Fare"      ,   "Embarked"  ,  "Title"       ,     "singlemale" , "womenchild")
tr1=tr1[,usefulvar]
te1=te1[,usefulvar]
te1=te1[,-1]
#####################################################
#stacking starts here.
#We will do a 2-fold cv to get predictions
#slice training data into 2.
set.seed(1234)
intrain1=createDataPartition(tr1$Survived,list=F)
train1=tr1[intrain1,]
train2=tr1[-intrain1,]
#do cv and get predictions on training set
pred1=rep(0,nrow(tr1))
pred2=pred1
pred3=pred1
pred4=pred1
pred5=pred1

#control method
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

#the only tuning parameter for rf is "mtry", # of variables selected at each node.
#we only have very few variables, so no need to tune it.
set.seed(1234)
fit1_1=train(Survived~.,data=train1,metric = "ROC",method="rf",trControl = cv.ctrl)
fit1_2=train(Survived~.,data=train2,metric = "ROC",method="rf",trControl = cv.ctrl)
pred1_1=predict(fit1_1,train2[,-1])
pred1_2=predict(fit1_2,train1[,-1])
pred1[intrain1]=pred1_2
pred1[-intrain1]=pred1_1
pred1=pred1-1
#fit 2 layer network
library(h2o)
## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', nthreads = -1) # use all CPUs (8 on my personal computer)
## MNIST data as H2O
train_h2o = as.h2o(train1)
## train model
set.seed(1234)
fit2_1 =
  h2o.deeplearning(x = 2:11,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs


test_h2o=as.h2o(train2[,-1])
## classify test set
h2o_y_test <- h2o.predict(fit2_1, test_h2o)
## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
pred2_1=df_y_test[,1]

## MNIST data as H2O
train_h2o = as.h2o(train2)
## train model
set.seed(1234)
fit2_2 =
  h2o.deeplearning(x = 2:11,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs


test_h2o=as.h2o(train1[,-1])
## classify test set
h2o_y_test <- h2o.predict(fit2_2, test_h2o)
## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
pred2_2=df_y_test[,1]

pred2[intrain1]=pred2_2
pred2[-intrain1]=pred2_1
pred2=pred2-1

## shut down virutal H2O cluster
h2o.shutdown(prompt = F)

#fit glm

set.seed(1234)
fit3_1=train(Survived~.,data=train1,method="glm",metric = "ROC", 
           trControl = cv.ctrl)
fit3_2=train(Survived~.,data=train2,method="glm",metric = "ROC", 
             trControl = cv.ctrl)
pred3_1=predict(fit3_1,newdata=train2[,-1])
pred3_2=predict(fit3_2,newdata=train1[,-1])
pred3[intrain1]=pred3_2
pred3[-intrain1]=pred3_1
pred3=pred3-1

#fit boosting
#adaboost has 3 tuning parameters: iter maxdepth and nu (shrikage parameter)
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))
set.seed(1234)
fit4_1 <- train(Survived~., 
              data = train1,
              method = "ada",
              metric = "ROC",
              tuneGrid = ada.grid,
              trControl = cv.ctrl)
fit4_2 <- train(Survived~., 
                data = train2,
                method = "ada",
                metric = "ROC",
                tuneGrid = ada.grid,
                trControl = cv.ctrl)
pred4_1=predict(fit4_1,train2[,-1])
pred4_2=predict(fit4_2,train1[,-1])
pred4[intrain1]=pred4_2
pred4[-intrain1]=pred4_1
pred4=pred4-1
#fit svm
set.seed(1234)
fit5_1 <- train(Survived~., 
              data = train1,
              method = "svmRadial",
              tuneLength = 9,#tune cost parameter. length=9 means we will test C=c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
              preProcess = c("center", "scale"),#normalize data as svm is sensitive to scale.normalization happens in each resample loop.
              metric = "ROC",
              trControl = cv.ctrl)
fit5_2 <- train(Survived~., 
                data = train2,
                method = "svmRadial",
                tuneLength = 9,#tune cost parameter. length=9 means we will test C=c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
                preProcess = c("center", "scale"),#normalize data as svm is sensitive to scale.normalization happens in each resample loop.
                metric = "ROC",
                trControl = cv.ctrl)
pred5_1=predict(fit5_1,train2[,-1])
pred5_2=predict(fit5_2,train1[,-1])
pred5[intrain1]=pred5_2
pred5[-intrain1]=pred5_1
pred5=pred5-1
#combine true y values from training set and 5 predictions into a data frame, fit a logistic regression (second-stage model)
dat=cbind(tr1$Survived,pred1,pred2,pred3,pred4,pred5)
dat[,1]=dat[,1]-1
colnames(dat)[1]="Survived"
dat=data.frame(dat)
dat[,1:6]=lapply(dat[,1:6],factor)
for(i in 1:6){
levels(dat[,i])=c("no","yes")
}
#second-stage model
superfit=train(as.factor(Survived)~.,data=dat,method="glm",metric = "ROC", 
                    trControl = cv.ctrl)
#also need 5 first-stage models fitted on full training data
set.seed(1234)
fit1=train(Survived~.,data=tr1,metric = "ROC",method="rf",trControl = cv.ctrl)

localH2O = h2o.init(max_mem_size = '6g', nthreads = -1) # use all CPUs (8 on my personal computer)
train_h2o = as.h2o(tr1)
set.seed(1234)
fit2 =
  h2o.deeplearning(x = 2:11,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                   balance_classes = TRUE, 
                   hidden = c(100,100), # two layers of 100 nodes
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, # use it for speed
                   epochs = 15) # no. of epochs

set.seed(1234)
fit3=train(Survived~.,data=tr1,method="glm",metric = "ROC", 
             trControl = cv.ctrl)

set.seed(1234)
fit4 <- train(Survived~., 
                data = tr1,
                method = "ada",
                metric = "ROC",
                tuneGrid = ada.grid,
                trControl = cv.ctrl)

set.seed(1234)
fit5 <- train(Survived~., 
                data = tr1,
                method = "svmRadial",
                tuneLength = 9,#tune cost parameter. length=9 means we will test C=c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
                preProcess = c("center", "scale"),#normalize data as svm is sensitive to scale.normalization happens in each resample loop.
                metric = "ROC",
                trControl = cv.ctrl)

#get predicitons on test set
preddf=data.frame(matrix(nrow=418,ncol=5))
preddf[,1]=predict(fit1,te1)

test_h2o=as.h2o(te1)
## classify test set
h2o_y_test <- h2o.predict(fit2, test_h2o)
## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
preddf[,2]=df_y_test[,1]

preddf[,3]=predict(fit3,te1)
preddf[,4]=predict(fit4,te1)
preddf[,5]=predict(fit5,te1)
colnames(preddf)=c("pred1","pred2","pred3","pred4","pred5")
#get final predictions with second-stage model, where covariates are predictions from the 5 first-stage model on test data
finalpred=predict(superfit,newdata=preddf)
Survived=as.numeric(finalpred)-1
PassengerId=te$PassengerId
result1=data.frame(PassengerId,Survived)
library(readr)
write_csv(result1,"result1.csv")
