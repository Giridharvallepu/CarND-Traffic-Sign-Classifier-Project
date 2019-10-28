# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Write-up-images/sign-images-plot.png "Sign Images plot"
[image2]: ./Write-up-images/Original-and-Preprocessed-image.png "Original and Preprocessed images"
[image3]: ./Write-up-images/1.jpg "Traffic Sign 1"
[image4]: ./Write-up-images/2.jpg "Traffic Sign 2"
[image5]: ./Write-up-images/3.jpg "Traffic Sign 3"
[image6]: ./Write-up-images/4.jpg "Traffic Sign 4"
[image7]: ./Write-up-images/5.jpg "Traffic Sign 5"
[image8]: ./Write-up-images/Test-Images-top-5-prob.png "Test Images Top 5 Probabilities"
[image9]: ./Write-up-images/feature-map-conv1.png "Feature map Conv1 layer"
[image10]: ./Write-up-images/feature-map-pool1.png "Feature max pool1 layer"
[image11]: ./Write-up-images/feature-map-conv2.png "Feature map Conv2 layer"
[image12]: ./Write-up-images/feature-map-pool2.png "Feature max pool2 layer"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and code is in Traffic_Sign_Classifier.ipynb

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I randomly choose 40 images from the training set and plot them here. We can see that there are some images much darker than others.


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used two techniques to preprocess the data set: 1. Grayscale 2. Mean Normalization.

As a first step, I decided to convert the images to grayscale because many images in the data test are dark and blurred. And the number of images may be not large enough to train a good model from scratch, so dropping the color information may help the model focus more on the edge of signs and the shape information.

As a last step, I normalized the image data because using mean normalization can make optimizer algorithm more easily minimize the loss function to train the model.

Here is the original traffice sign images and preprocessed images(after grayscaling and normalization).

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented the LeNet-5 neural network architecture for classifying the traffic signs.

My final model consisted of the following layers and can found in Jupyter notebook cell under Model Architecture

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6			        |
| Convolution 5x5       | 1x1 stride, valid padding,outputs 10x10x16    |
| RELU					|												|
| Max pooling	 		| 2x2 stride,  outputs 14x14x6					|
| FLATTEN 				| Outputs 400       							|
| Dropout				|	dropout_keep_probability:0.5				|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				|	dropout_keep_probability:0.5				|
| Fully connected		| outputs 84   									|
| RELU					|												|
| Dropout				|	dropout_keep_probability:0.5				|
| Fully connected		| outputs 43   									|
| softmax        		| predicts class probability					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with 0.0005 as the learning rate and number of epochs is 200, the batch size is 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.966
* test set accuracy of 0.959

If a well known architecture was chosen:
* What architecture was chosen?
 I had chosen the popular LeNet-5 neural network architecture for classifying the traffic signs
 
* Why did you believe it would be relevant to the traffic sign application?
Because I had used the same one for classifying hand written digits using MNIST database and it's very accurate in classifying the test data. And also its very popular one thats developed by Yann Lecun.

Since traffic signs classifications looks similar to handwritten digits classification, I implemented LeNet architecture with additional dropout step.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 I achieved the training set accuracy of 0.995, the validation set accuracy of 0.966, the test set accuracy of 0.959 and this confirms  thats this model is working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution 		| General caution								| 
| Road work    			| Road work 									|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Keep right      		| Keep right					 				|
| Speed limit (60km/h)	| Speed limit (60km/h) 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th,22nd, 23rd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a General caution (probability of 0.999999), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999999      			| General caution								| 
| .000001 				| Road work 									|
| .000000				| pedestrians         							|
| .000000      			| Go staringht of left							|
| .000000			    | Right-of-way at the next intersection			|


For the second image, the model is relatively sure that this is a Road work(probability of 0.989488), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .989488 				| Road work 									| 
| .003823 				| Dangerous curve to the right					|
| .001459				| General caution								|
| .001409     			| Bunmpy Road   				 				|
| .001311			    | Traffic signals      							| 

For the third image, the model is relatively sure that this is a Speed limit (30km/h)(probability of 0.989488), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999949 				| Speed limit (30km/h)							| 
| .000050 				| Speed limit (50km/h)							|
| .000001				| Speed limit (20km/h)							|
| .000000     			| Speed limit (70km/h)							|
| .000000			    | Speed limit (80km/h)							| 

For the fourth image, the model is relatively sure that this is a Keep right(probability of 1.000000), and the image does contain a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000000 				| Keep right            						| 
| .000000 				| Speed limit (20km/h)							|
| .000000				| Speed limit (30km/h)							|
| .000000     			| Speed limit (50km/h)							|
| .000000			    | Speed limit (60km/h)							| 

For the firth image, the model is relatively sure that this is a Speed limit (60km/h)(probability of 0.998731), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998731 				| Speed limit (60km/h)							| 
| .001241 				| Speed limit (80km/h)							|
| .000026				| Speed limit (50km/h)							|
| .000002     			| Speed limit (30km/h)							|
| .000000			    | Dangerous curve to the right					| 

The image below is the visualization of how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

![alt text][image8]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I used the outputFeatureMap() function to visualize the LeNet-5 model's convolutional layers' and max pooling layers' feature map.

General caution sign image of the new images is the input stimuli image.

Here are the feature maps of convolutional layer 1. we can see observe that the network is mainly focusing on the edges of the triangular signs and the inner exclamation mark.

![alt text][image9]

Here are the feature maps of max pooling layer 1.We can see that the max pooling operation reduce the size of the input, and allow the neural network to focus on only the most important elements, in this case, the edges of the signs and the inner exclamation mark.
![alt text][image10]


Here are the feature maps of convolutional layer 2.

![alt text][image11]

Here are the feature maps of max pooling layer 2:

![alt text][image12]

Looks very hard to determine what the neural network is focusing on in convolutional layer 2 and max pooling layer 2. In these layers, the model is focusing on more abstract and high-level information.
