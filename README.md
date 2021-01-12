# pytorch-linear-models
Implementation of linear models (OLS/LASSO/Ridge) in base PyTorch (so no torch.nn). This repo tries to follow the sklearn API for easy integration.

# Useage
```
from regression import *

clf = LinearRegression(penalty=None) #Penalty can be one of: None for  OLS, 'l1' for LASSO or 'l2' for Ridge
clf.fit(X_train,y_train) # fit the model like any sklearn model

clf.predict(X_train) # make predictions on new data

clf.plot_history() # Plot loss over time
``` 

# Requirements:
torch
numpy
seaborn

#  Work in progress:
Working on itegrating (regularized) logistic regression.
