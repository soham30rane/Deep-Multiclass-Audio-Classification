# Notes for week 1 of Course 4

## Case studies

Few clasic Networks

* LeNet-5
* AlexNet
* VGG
* **ResNet**
* **Inception**

**LeNet-5**
![alt text](imgs/image-15.png)

* It was developed to classify handwritten digits
* It had 60 thousand parameters, which significatnly smaller compared to modern networks
* Didn't use RelU activation, instead used sigmoid and tanh
* Did not use Max pooling instead used Average Pooling
* Did not used softmax classifier, used some different classifier

* **AlexNet**
![alt text](imgs/image-16.png)

* It was developed for classeify 1000 classes of objects
* It has 6 million parameters
* Since at the time of release, the Computational power wasn't very promising, it used multiple GPUs to train.

* **AlexNet**
![alt text](imgs/image-17.png)

* Simple architecture
* Always used Convolution layers of 3 by 3 filters with same padding
* Always used Max Pooling layers of 2 by 2 with stride of 2
* It had total of 16 layers which had learnable weights
* Has 135 Million parameters
* It uniformity made it a attractive model
* The general observation was the dimensions went on decreasing and the number of channels went on increasing

## ResNets

Deep Neural Networks are difficult to train because it results in Vanishing / Exploding gradients
This is where skip connections come into play. Basically you take the output of a layer and feed in as input to more deeper layer into the network
This helps us build a very deeper networks.

Below is an example of Normal Network
![alt text](imgs/image-18.png)

Below is an Example of skip connection
![alt text](imgs/image-19.png)

* The A[l] is injected just before the ReLU part
* This allows us to train much deeper Neural Network
* The 2 layers in the skip connections need to have same dimensions. Hence "same" padding is used mostly.
* If the dimenions are not same then an extra weight matrix is added.
For example here, since A[l] has 128 units and A[l+2] has 256 units, we add a weight matrix of (256 by 128) so that when we take matrix product of (W x A[l]) we get 256 by 1 output
![alt text](imgs/image-23.png)

**Residual Block**
![alt text](imgs/image-20.png)
![alt text](imgs/image-21.png)
![alt text](imgs/image-22.png)

![alt text](imgs/image-24.png)

**1 by 1 Convolution**
![alt text](imgs/image-25.png)
Its useful when you have to only change the number of channels in any layer.
For example the above image has a layer with 32 channels. We could use 16 filters of 1 by 1 to convert the number of channels to 16

## Inception Network (GoogleNet)

* Instead of picking one of (1,1) / (3,3) / (5,5) or some other convolution, in inception network we can pick multiple of those at the same time what whatever the outputs are, stack them on top of each other (for this "same" padding has to be applied)
![alt text](imgs/image-26.png)
Note in the above image for Pooling to work we need have stride of 1 and have same padding

![alt text](imgs/image-29.png)
The above Convolution requires a whopping 120 million multiplication
This is where 1 by 1 Convolutions can help us
![alt text](imgs/image-30.png)
This reduced the computional cost by a tenth upto 12 million multiplications

These were the building blocks of the Inception Network

Below is a representation of one inception module
![alt text](imgs/image-31.png)
An inception Network is made by putting together a lot of such modules together and is shown below
![alt text](imgs/image-32.png)
One detail that is missing in the above image is that there also exist a side branches.
![alt text](imgs/image-33.png)
These side branches ensure that the features learnt till that particular hidden layer are also good enough to make prediction

## MobileNet

* Allows to build and run Neural Networks that would also work in low compute environment like mobile phone
* Useful for Mobile and embedded applications
* Key idea is that they have **dephtwise seperable convolutions** instead of normal convolution

![alt text](imgs/image-34.png)
Note the computational cost

![alt text](imgs/image-35.png)
In here, one filter of size just 3 by 3  is used only on one channel for example Red. Thats the reason its called Depthwise Convolution. This way we need just 3 filters of size of just 3 by 3 to get the output.
Note that the output and input ends up having same number of dimensions.
Hence to get the desired number of outputs, pointwise convolution is applied, which carry out 1 by 1 convolutions to get the desired number of channels
![alt text](imgs/image-36.png)
![alt text](imgs/image-37.png)
![alt text](imgs/image-38.png)
These were the building blocks for MobileNet

Have a look at MobileNet V1 and V2 architectures.
![alt text](imgs/image-39.png)
These have a significantly low computation costs
V2 architecture is different from V1 in following ways

* The single block is repeated 17 times in V2 where as 13 times in V1
* V2 contains Residual Connections
* V2 has one extra operation in each block called "Expansion" operation.
![alt text](imgs/image-40.png)
The advantage of the bottleneck block is that it required less memory while passing the activations to the next block and hence is more efficent memorywise

## EfficientNet

* So MobileNets are computationally less expensive but come with a bit of sacrifice in a performance
* So if we are in a Computationally limited environment then we need our Model to be less computationally expensive at the expense of accuracy, otherwise we could allow it to be computationally expensive and give us a better accuracy.
* EfficientNet allows a model to automatically perform according to the environment

![alt text](imgs/image-41.png)
So there are basically 3 things to you can do to get better performance with any model as stated above

* Use high resolution input
* Use deeper Network
* Increase the width of layers. (No. of units or dimensions)

So one of these things is picked depending on the enironment

## Transfer Learning

Transfer learning refers to using the parameters of a model which has already been trained on some data. These parameters are used as a good initialization point for our model.
Instead of initialization we could also freeze some parameters (assuming they are well learnt features) and use the rest of the model
![alt text](imgs/image-42.png)

## Data Augmentation

Creating more data (by subtle variations of existing data) from the existing data.
![alt text](imgs/image-43.png)
![alt text](imgs/image-44.png)

**State of Computer Vision**
![alt text](imgs/image-45.png)
![alt text](imgs/image-46.png)
