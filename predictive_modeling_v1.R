
# Opening training and test data
trdata <- read.csv("pml-training.csv",stringsAsFactors=FALSE,na.strings = "NA")
testdata <- read.csv("pml-testing.csv",stringsAsFactors=FALSE,na.strings = "NA")

# Combine both training and test data to perform pre-processing at once
# The following pre-processing will be performed
# 1) Drop variables from analysis if there is more than xxx percentage of NA value present
# 2) If there are any NA value presents then we will impute with median for numeric variable and mode for categorical variables

# Dropping problem_id variable from test data
testdata <- testdata[,-which(names(testdata)=="problem_id")]

# Adding the classe variable in test data with class X, we will remove this after pre processing step
testdata$classe<-"X"

# Adding new variable to track training and test data
trdata$traintest <- 1
testdata$traintest <-0

# Combining both dataset
alldata <- rbind(trdata,testdata,stringsAsFactors=F)

# Calculating percentage of NA values in a variable
napct <- apply(alldata,2,function(x) 100*sum(is.na(x))/length(x))
blankpct <- apply(alldata,2,function(x) 100*sum(nchar(x)==0)/length(x))

# Index of the variable that will be dropped due to excess NA values
dropvarIndex <- c(which(napct>10),which(blankpct>10))
length(dropvarIndex)

# dropping the variables with more than 10% NA values
alldata <- alldata[,-dropvarIndex]

# dropping categorical variables with very sparse data
table(alldata$new_window)
alldata <- alldata[,-which(names(alldata)=="new_window")]

# Dropping the variable with name X because this is just the serial number from the csv file
alldata <- alldata[,-which(names(alldata)=="X")]

# Checking distribution of classe variable
table(alldata$classe)

# Dropping observation with classe==FALSE
alldata <- alldata[-which(alldata$classe=="FALSE"),]

# Now separating out the training and test data

trainingData <- alldata[alldata$traintest==1,]
trainingData <- trainingData[,-which(names(trainingData)=="traintest")]

testData <- alldata[alldata$traintest==0,]
testData <- testData[,-which(names(testData)=="traintest")]
testData <- testData[,-which(names(testData)=="classe")]

# extracting only the variable names that will be used in modeling stage
modelingVar <- names(trainingData)[-c(1:4)]

# converting the outcome variable into factor mode for analysis
trainingData$classe <- as.factor(trainingData$classe)

# Creating training and validation data by splitting initial training data
library(caret)
set.seed(4929)
inTrain <- createDataPartition(trainingData$classe, p = 3/4)[[1]]
training <- trainingData[ inTrain,]
validation <- trainingData[-inTrain,]

# Model building

# Setting model control parameter with 10-fold cross validation
ctrl <- trainControl(method = "cv",number=10)

# Fitting the knn model with k=5, though we have used different k values (5,7,9,11,13) but in we kept only the best one with tuneLength=1
model_1 <- train(classe~.,data=training[modelingVar],method="knn",tuneLength = 1,trControl = ctrl)

# printing model output
model_1

# Predicting on validation data
predict_1 <- predict(model_1,newdata=validation[modelingVar[-length(modelingVar)]])

# calculating accuracy statistics based on the prediction on validation data
confusionMatrix(data=predict_1,reference=validation$classe)

# Prediction on test data
predict_test <- predict(model_1,newdata=testData[modelingVar[-length(modelingVar)]])

# preparing test data with predicted class levels
predicted_test_data <- data.frame(testData,classe=predict_test)
write.csv(predicted_test_data,file="predicted_test_data.csv",row.names=F)
