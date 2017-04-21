## Result and Conclusion

Overall, we examine different models on our data and we get some nice result. The most distinguishing features of voice data are mean fundamental frequency and the interquantile range (IQR). In general, males tend to have a lower mean frequency of voice and their voice's IQR is higher. Some other helpful feature includes the standard deviation of voice and first quantile of voice. 

In terms of the models, we tried quite a few. The outstanding ones are logistic regression, KNN, decision tree and random forest. We use variable selection in logistic regression to get a thorough model with accuracy of 97%. For the supervised machine learning models, decision tree has the fastest learning rate, with an accuracy of 95% to 96%. For KNN model, we did two versions of it. One with only two features, meanfun & IQR, and the other using a 10-fold validation with all the features. Both model presented accuracy higher than 96%, so we choose to present the simpler one. 

In the process of builing and training our models, we use the knowledge and technique we learn in the lab. Moreover, we also learn online for ways to generate nice and representative plots for different models. 


