# pytorch-linear-models
Implementation of linear models (OLS/LASSO/Ridge) in base PyTorch (so no torch.nn). This repo tries to follow the sklearn API for easy integration with existing projects.

## Usage
```
from regression import LinearRegression

clf = LinearRegression(penalty=None)  # Penalty can be one of: None for  OLS, 'l1' for LASSO or 'l2' for Ridge

clf.fit(X_train,y_train)              # Fit the model like any sklearn model

clf.plot_history()                    # Plot loss over time

clf.predict(X_test)                   # Make predictions on new data


``` 

## Requirements:
* torch==1.7.0
* numpy==1.18.5
* seaborn==0.10.0

## Work in progress:
Working on integrating (regularized) logistic regression.
