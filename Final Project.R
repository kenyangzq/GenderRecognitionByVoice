## Final Project
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
mf.table <- table(df.prediction$predict, df.prediction$actual)
mf.table
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
# let female be postive, true positive rate is 1541/1584 = 0.9728

### KNN






