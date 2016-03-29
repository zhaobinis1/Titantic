tr=read.csv("train.csv")
te=read.csv("test.csv")
gcm=read.csv("genderclassmodel.csv")
gm=read.csv("gendermodel.csv")

library(randomForest)



#check na in train and test
sapply(te, function(x) sum(is.na(x)))#86 na in age. 1 na in fare
sapply(tr, function(x) sum(is.na(x)))#177 na in age.
#check association of missing age with survived
tr2=tr1
tr2$na.age=is.na(tr2$Age)
pairs(tr2)
chisq.test(table(tr2$na.age,tr2$Survived))

#combine two sets
library(dplyr)
full=bind_rows(tr,te)
# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare'

# Show title counts by sex again
table(full$Sex, full$Title)
# Finally, grab surname from passenger name
full$Surname=gsub(",.*","",full$Name)

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable 
full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
library(ggplot2)
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='bin', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size')

#62 and 830 of embarkment is missing 
which(full$Embarked=="")

#check class, fare and embarkment
ggplot(full[full$Embarked!="",], aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot()

#check fare and class for those two missing embarkment
full[full$Embarked=="",]#80 dollars 1st class, so very likely have embark=c
full$Embarked[full$Embarked==""]="C"

#1 missing fare
full[1044, ]
#This is a third class passenger who departed from Southampton (‘S’). 
#Let’s visualize Fares among all others sharing their class and embarkment (n = 494).
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1)
#replace with median
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



#fit rf
library(caret)
#control method
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
#the only tuning parameter for rf is "mtry", # of variables selected at each node.
#we only have very few variables, so no need to tune it.
set.seed(1234)
fit1=train(Survived~.,data=tr1,metric = "ROC",method="rf",trControl = cv.ctrl)
pred1=predict(fit1,te1)
#fit 2 layer network
library(h2o)
## start a local h2o cluster
localH2O = h2o.init(max_mem_size = '6g', nthreads = -1) # use all CPUs (8 on my personal computer)
## MNIST data as H2O
train_h2o = as.h2o(tr1)
## train model
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
#After training the model we can look at the confusion matrix. 
# print confusion matrix
h2o.confusionMatrix(fit2)
#Now, let’s predict the test data and save the results.
test_h2o=as.h2o(te1)
## classify test set
h2o_y_test <- h2o.predict(fit2, test_h2o)
## convert H2O format into data frame and  save as csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
## shut down virutal H2O cluster
h2o.shutdown(prompt = F)

#fit glm
library(caret)
set.seed(1234)
fit3=train(Survived~.,data=tr1,method="glm",metric = "ROC", 
           trControl = cv.ctrl)
pred3=predict(fit3,newdata=te1)

#fit boosting
#adaboost has 3 tuning parameters: iter maxdepth and nu (shrikage parameter)
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))
set.seed(1234)
fit4 <- train(Survived~., 
                  data = tr1,
                  method = "ada",
                  metric = "ROC",
                  tuneGrid = ada.grid,
                  trControl = cv.ctrl)
pred4=predict(fit4,te1)
#fit svm
set.seed(1234)
fit5 <- train(Survived~., 
                  data = tr1,
                  method = "svmRadial",
                  tuneLength = 9,#tune cost parameter. length=9 means we will test C=c(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
                  preProcess = c("center", "scale"),#normalize data as svm is sensitive to scale.normalization happens in each resample loop.
                  metric = "ROC",
                  trControl = cv.ctrl)
pred5=predict(fit5,te1)
#compute final prediction
Survivedlist=list()
Survivedlist[[1]]=as.numeric(pred1)-1
Survivedlist[[2]]=as.numeric(df_y_test[,2])-1
Survivedlist[[3]]=as.numeric(pred3)-1
Survivedlist[[4]]=as.numeric(pred4)-1
Survivedlist[[5]]=as.numeric(pred5)-1
Survivedlist=do.call(cbind,Survivedlist)
#mode function
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
Survived=apply(Survivedlist,1,Mode)
PassengerId=te$PassengerId
result1=data.frame(PassengerId,Survived)
library(readr)
write_csv(result1,"result1.csv")
