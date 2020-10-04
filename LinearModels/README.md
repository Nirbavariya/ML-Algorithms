# Linear Models 

Here, I have 3 algorithms:

### 1. Linear Regression using pseudo inverse
Usage - Use simply like a scikit-learn model. Methods available are *fit* and *predict*.


### 2. Linear Regression using Gradient Descent

This model uses Batch Gradient Descent algorithm to find the slope and intercept. Moreover, you can regularize this model. I have implemented the Elastic Net method.
#### hyperparameters

| name | default value | Description |
| --- | --- | --- |
| **n_iterations** | 2000 |  Number of iterations. | 
| **learning_rate** | 0.01 | ... | 
| **alpha** | 0 | For regularization. No regularization by default. |
| **r** | 0.5 | L1 ratio. When r = 0, The model will be equivalent to Ridge Regression(L2) and for r = 1, it will be equivalent to Lasso Regression(L1). |


#### Attributes
| name | description |
| --- | --- |
| **w** | weights (slope) |
| **b** | intercept |
| **history** | list of costs per iteration |

Usage - Use simply like a scikit-learn model. Methods available are *fit* and *predict*.


### 3. Logistic Regression

Usage - Use simply like a scikit-learn model. Methods available are *fit* and *predict*.

Note: As of now, this model may overfit because of no regularization parameters. Also, it can be used for binary classification only.