# **Traffic Sign Recognition | Writeup | Mohit Arvind Khakharia** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./custom_pics/1.png "Traffic Sign 1"
[image2]: ./custom_pics/2.png "Traffic Sign 2"
[image3]: ./custom_pics/3.png "Traffic Sign 3"
[image4]: ./custom_pics/4.png "Traffic Sign 4"
[image5]: ./custom_pics/5.png "Traffic Sign 5"
[image6]: ./custom_pics/5.png "Traffic Sign 5"
[image7]: ./images_for_writeup/custom_image_processing.png"Custom_image_processing"
[image8]: ./images_for_writeup/custom_images.png "Custom_images"
[image9]: ./images_for_writeup/dataset_augmented.png "Dataset_augmented"
[image10]: ./images_for_writeup/frequency_analysis.png "Frequency_analysis"
[image11]: ./images_for_writeup/grayscale_conversion.png "Grayscale_conversion"
[image12]: ./images_for_writeup/Lenet.png "Lenet"
[image13]: ./images_for_writeup/optimizers.png "Optimizers"

---
### Project Code

[project code](https://github.com/Mohit-Ak/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

### Before Augmentation
* Size and shape of the training set
```
Size - 34799 Samples
Shape - 34799 x 32 x 32 x 3

 ```
* Size and shape of the Vlidation set
```
Size - 4410 Samples
Shape - 4410 x 32 x 32 x 3
```
* Size and shape of the test set
```
Size - 12630 Samples
Shape - 126302 x 32 x 3
```

* Shape of a Traffic Sign image
``` 
32 x 32
```
* Unique Classes/Labels in the data set
```
43
```

### Problem Aspects considered

- How to avoid over or underfitting?
- Are techniques like normalization, rgb to grayscale, shuffling needed
- Number of examples per label (some have more than others).
- Should we generate fake data / augmentation.
The decisions taken are described below.

## Data Augmentation

As mentioned in the lecutre, it is always a good practice to augment the data before training inorder to achieve translationaly, rotational and brightness invariance.

### Trasnformations Applied
- Rotation
- Shear
- Zoom
- Translation
### Dataset size increase
 - ```Training Data new size = 69598 | Approximately 2x```
 - ```Validation Data new size = 17640 | Approximately 1.5x```
 
### Visualization of the Augmented dataset.

![alt text][image9]

### Statistical Analysis of the Class frequency in the Augmented data
- Tells us if we have enough samples of all the classes
- Tells us if our training data is not biased
- Tells us that if we cover all the classes in the Validation set
- Tells us if we test all the classes in the Test set

![alt text][image10]

## Model Architecture
- The architecture implemented was a modified version of LeNet-5 shown in the  [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81).

### Preprocessing the augmented dataset

- Normalized the data has mean zero and equal variance by using the following formula - `(pixel - 128)/ 128`
- Converted to grayscale by averaging the channels.
- Shuffled the data for preventing a biased learning curve when using batch based Gradient algorithms as they assume the batch is an approximation of the entire dataset.

```
Training Data Shape - Before Preprocessing
(69598, 32, 32, 3)
Training Data Shape - After Preprocessing
(69598, 32, 32, 1)
Validation Data Shape - Before Preprocessing
(17640, 32, 32, 3)
Validation Data Shape - After Preprocessing
(17640, 32, 32, 1)
Testing Data Shape - Before Preprocessing
(12630, 32, 32, 1)
Testing Data Shape - After Preprocessing
(12630, 32, 32, 3)
```

![alt text][image11]

### REASON
- Converting to grayscale -It helped to reduce training time and makes detection color agnostic.
- Normalizing the data to the range (-1,1) - As mentioned in the classes, a wider distribution in the data would make it more difficult to train using a singlar learning rate. Also, the math becomes difficult at extremely large [or] extremenly small numbers.

#### Original LeNet Model Architecture
![alt text][image12]

The architecture I used was a modified version of LeNet as it already did a preety good job of classifying images and need a few tweaks to get exceptional performance.

Custom Modified Architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												| outputs 28x28x6
| Max pooling	      	| 1 2 2 1 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | outputs 10x10x16     									|
| RELU					|												| outputs 10x10x16  
| Max pooling	      	| 1 2 2 1 stride,  outputs 5x5x16  				|
| Flattening	      	| output 400  				|
| Fully connected		| output 200        									|
| Fully connected		| output 100        									|
| Softmax				| Result output 43        									|


## Training the model
### Hyper Parameters and experimentation
- EPOCHs = 100 - Tried with other epochs but it did not imporve the accuracy a lot. Higher values caused overfitting and small values caused under fitting.
- LEARNING RATE = 0.001 - After experimenting it was realized that 0.001 is a really good learning rate for 100 epochs.
- MU = 0
- SIGMA = 0.1 - Gives a really good normal distributon for choosing the random weights.
- KERNEL SIZE = 5 - Experimentation proved that it was a good size for an image of size 32 x 32
- BATCH SIZE = 128 - Tried with higher batch sizes but only resulted in more memory and time without any noticable increase in accuracy.
- KEEP PROBABILITY = 1.0 - This method is used more of a replacement to max-pooling now days but since I already had  max-pooling doing the job for me, I did not face the need to change the "Keep Probabilty" to lower numbers.

### Optimizer
*AdamOptimizer*
- This works really well because this method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
- Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods.
![alt text][image13]

### Fancy Stuff experimented
- Tried to implement Hypertuning. An approach to autmatically find the best set of parameters.
- Played around with Tensor board and TensorFlow summaries for visualization.
- Played around with a capsule network solution for image classification.

#### FINAL ACCURACY
Accuracy | Validation Dataset
Epoch : 84
Accuracy : 94.6%
Epoch : 100
Accuracy : 94.1%
Accuracy | Testing Dataset
Accuracy : 92.7% 

### Model on Custom Images

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


