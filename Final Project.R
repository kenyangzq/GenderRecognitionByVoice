## Final Project
# Notice: need to set the directory to your directory that contains the data file
library(ggplot2)
library(dplyr)
library(broom)
library(class)

voice <- read.csv("voice.csv")
summary(voice)
View(voice)

### Pre-processing data
# Mode, dfrange and modindex contains several 0's, may need to mark them as
# missing data. 
# However, on Kaggle, people are working on this data without doing this. The
# source of the data claim that the data has been pre-processed
voice.sort_by_dfrange <- voice %>% arrange(dfrange,mode)
View(voice.sort_by_dfrange)

# Label is gender, a binomial variable, can't use linear regression on it.




### Logistic Regression 
## using only mean frequency to predict gender
fit.mf <- glm(label~meanfreq, data = voice, family = "binomial")

# get the prediction and mutate it into binary output
mf.prob.pred <- predict(fit.mf, newdata = voice, type = "response")
mf.outcome.pred <- ifelse(mf.prob.pred > .5, 
                          "predict male", 
                          "predict female")

mf.df.prediction <- data.frame(
  predict = mf.outcome.pred,
  actual = voice$label
)

# get the prediction table
mf.table <- table(mf.df.prediction$predict, mf.df.prediction$actual)
mf.table
### female male
### predict female   1101  648
### predict male      483  936
# not very satifactory with tf(true positive) = 1101/1584 = 0.6951


## predict gender using variable selection
fit.all <- glm(label~., data = voice, family="binomial")
fit.none <- glm(label~1, data = voice, family="binomial")
summary(fit.all)

fit.result <- step(fit.none, 
                   scope = list(lower = fit.none,
                                upper = fit.all),
                   direction = "forward")

all.prob.pred <- predict(fit.result, newdata = voice, type = "response")
all.outcome.pred <- ifelse(all.prob.pred > .5, 
                           "predict male", 
                           "predict female")

all.df.prediction <- data.frame(
  predict = all.outcome.pred,
  actual = voice$label
)

all.table <- table(all.df.prediction$predict, all.df.prediction$actual)
all.table
### female male
### predict female   1541   36
### predict male       43 1548
# let female be postive, true positive rate is 1541/1584 = 0.9728






### KNN
# I did an experiment with meanfreq and sd. It turns out not very 
# well. I will try to work with different combination of parameters

# first break the data set into train set and test set

voice.sub <- voice %>% select(meanfreq, sd, label)

row <- nrow(voice)
set.seed(1234)
voice.train <- voice.sub %>% sample_n(row/10*9)
voice.test <- setdiff(voice.sub, voice.train)

# then get data without label and separate label out
train.data <- voice.train %>% select(-label)
test.data <- voice.test %>% select(-label)
train.label <- voice.train$label
test.label <- voice.test$label

# perform KNN with K = 5
knn.pred.5 <- knn(train.data, test.data, train.label, 5)
table(knn.pred.5, test.label)

# Plot KNN:
summary(voice.sub)

test <- test.data %>% mutate(
  pred5 = knn.pred.5
)

grid <- expand.grid(
  meanfreq = seq(0, 0.3, length.out = 100),
  sd = seq(0, 0.3, length.out = 100)
)

set.seed(1234)
grid5.pred <- knn(train.data, grid, train.label, 5)

ggplot(test, aes(x=meanfreq, y=sd)) +
  geom_point(aes(pch=pred5, color = pred5), size = 3) +
  geom_point(data = grid, mapping = aes(x=meanfreq,y=sd,color=grid5.pred), 
             alpha = .2) + 
  ggtitle("K=5")

# Not very informative from the graph, maybe other parameter choices are better




## Cross Validation

# Create data partition
set.seed(4321)
trainindex <- createDataPartition(
  y = voice$label,
  p = .8,
  list = FALSE
)
length(trainindex)

cv_train <- voice[trainindex, ]
cv_test <- voice[-trainindex, ]

# create control variable
control_var <- caret::trainControl(
  method = "cv",
  number = 5
)

# let's try 1-layer neural network
install.packages("e1071")
set.seed(1)
nnet_fit <- caret::train(
  label ~ .,
  data = cv_train,
  method = "nnet",
  trControl = control_var
)
nnet_fit
# accuracy 97.5%, nice model



