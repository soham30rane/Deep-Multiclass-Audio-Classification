# Notes for week 2 of Course 1

## Logistic Regression as Neuaral Network

Logistic Regression is used for binary classification problems. The output is a binary variable which is either 0 or 1

![alt text](imgs/image.png)

The Sigmoid function (as given below) is applied to the output (Z) in case of binary classifictaion

![alt text](imgs/image-1.png)

**Loss Function**  
Calculates how close is the predicted output to the original output.  
Loss function is defined for a single example
**Cost Function**  
It is generally the average of loss function over many examples. The models aim is to minimize this cost function

**Loss function used for Binary classificatiion**  

![alt text](imgs/image-2.png)

**Cost function for Binary Classification**  

![alt text](imgs/image-3.png)

**Gradient Descent**  

![alt text](imgs/image-4.png)

![alt text](imgs/image-5.png)

**Computation Graph**  
Computation Graph organises the steps in which you compute any given function. For example
![alt text](imgs/image-6.png)
Its helps us calculate a derivatives in a simple manner by backpropogating from final output variable towards the input variables (right to left here)

![alt text](imgs/image-7.png)

The following slide represent one step of gradient descent in great detail  
![alt text](imgs/image-8.png)

Multiple for loops in the code makes it inefficient to work with large data. This results in a need for Vectorization.  

![alt text](imgs/image-9.png)  

Here is a one step of gradient descent in completely Vectorized form (without any for loops) shown in a great detail  
![alt text](imgs/image-10.png)  

**Python/Numpy tip**  
Don't use rank 1 arrays like these  

    #AVOID
    a = np.random.randn(5)  # This is a rank 1 array
    a.shape()       # Output is (5,)

Always specify complete dimensions

    a = np.random.randn(5,1)  # This is a rank 1 array
    a.shape()       # Output is (5,1)

Use asserts to make sure the shape is as expected

    assert(s.shape() == (5,1))
