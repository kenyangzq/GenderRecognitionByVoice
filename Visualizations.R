#This file generates some graphics for exploratory data analysis 

setwd("~/Github/GenderRecognitionByVoice")

library(ggplot2)
library(dplyr)
library(tidyr)

#setwd and load data
voice <- read.csv("voice.csv",stringsAsFactors = FALSE)
voice <- voice %>% mutate(label = factor(label))

#Code from a Kaggle notebook
#Density plots to help determine which features are important 
#png('imgs/Density_1.png', width = 1080, height = 720,res = 125)

voice %>% na.omit() %>%
  gather(type,value,1:20) %>%
  ggplot(aes(x=value,fill=label))+
      geom_density(alpha=0.3)+
      facet_wrap(~type,scales="free")+
      theme(axis.text.x = element_text(angle = 90,vjust=1), plot.title = element_text(size = 20,face = "bold", hjust = 0.5))+
      labs(title="Density Plots of Data across Variables")

#dev.off()

# Scatterplot of meanfun vs. IQR (two important features)
#png('imgs/Scatterplot_1.png',width = 1080, height = 720, res = 125)
ggplot(voice, aes(y = meanfun, x = IQR, colour = label)) + 
  geom_point()+
  labs(title = "Meanfun vs. IQR")+
  theme(plot.title = element_text(face = "bold",size = 18,hjust = 0.5),text = element_text(size=14))
#dev.off()




