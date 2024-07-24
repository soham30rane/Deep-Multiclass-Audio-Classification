# Notes for week 3 of Course 2

## Hyperparameter tuning

Here are some important hyperparameters
![alt text](imgs/image-33.png)

* Learning rate generally is the most important one
* Then 2nd priority is given to those boxed in yellow
* 3rd priority is given to those in purple

![alt text](imgs/image-34.png)
(left is grid and right has random points)
![alt text](imgs/image-35.png)
It basically implies to zoom in to the region where we are getting better results

**Using appropriate scale to pick random hyperparameters**
First evaluate the range of the target hyperparameter
If the range is too high or too small, then take random values on log scale, otherwise taking random values on linear scale works

![alt text](imgs/image-36.png)
There are two typical approaches for tuning hyperparameters

* You have enough computational resources to train the model in parellel for finding the best hyperparameters. This one is shown in the right

* You don't have enough computational resources, hence you keep developing the same model day by day. Keep experimenting with hyperparameters on the same model. This one is shown in the left

**Batch Normalization**
Previously we discussed normalization of input X
Similarly applying normalization between the hidden layers would also improve the model
Mostly the normalization is applied on Z instead of activations.
While we apply normalization we basically tranform our data to have a mean of 0 and variance of 1. But we could actually let the model learn what should be the best mean and variance in the normalization layer. This adds two learnable parameters
![alt text](imgs/image-37.png)
![alt text](imgs/image-39.png)

**Softmax Regression**
It used used in classification problems. In output layer, instead of sigmoid activateion, the softmax activation is used
![alt text](imgs/image-38.png)

**Deep learning Frameworks**
![alt text](imgs/image-40.png)
