
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2017_12_24_08_29_51_739.jpg "Center Image"
[image3]: ./examples/center_2017_12_24_08_29_51_739_flipped.jpg "Recovery Image"
[image4]: ./examples/center_2017_12_24_08_29_51_173_re.jpg "Center Re"
[image5]: ./examples/left_2017_12_24_08_29_51_173_re.jpg "Left Re"
[image6]: ./examples/right_2017_12_24_08_29_51_173_re.jpg "Right Re"
[image7]: ./examples/center_2017_12_24_08_29_51_173_cropp.jpg "Mid Cropp"
[image8]: ./examples/Model_Architecture.png "Model Architecture"
[image9]: ./examples/nvidia.png "nvidia net"
[image10]: ./examples/left_2017_12_24_08_29_51_173_cropp.jpg "left Cropp"
[image11]: ./examples/right_2017_12_24_08_29_51_173_cropp.jpg "right Cropp"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of two convolution neural network with 5x5 filter sizes and depths between 24 and 36 (model.py lines 59-60) 

The model includes RELU layers to introduce nonlinearity (code line 59-60), and the data is normalized in the model using a Keras lambda layer (code line 56). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 67). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , red lane cornering etc. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to giving enougth training data.

My first step was to use a convolution neural network model similar to the NVIDIA Architecture I thought this model might be appropriate because it's been proven by a real car!
![Nvidia][image9]
The layers of Nvidia net has 5 convolution layers, however, it's too big for our project, so I change it into 3 convolution layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. The speed is too slow, so I change the speed limit in `drive.py` to 25 MPH.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/jSaAvgxrW1Q/0.jpg)](https://www.youtube.com/watch?v=jSaAvgxrW1Q)

#### 2. Final Model Architecture

The final model architecture (model.py lines 56-64) consisted of a convolution neural network with the following layers and layer sizes:
2D Convolution(5 x 5 filter, 2 x 2 sub-sample) + 2D Convolution(5 x 5 filter, 2 x 2 sub-sample) + Flatten+Dense(60)+Dense(42)+Dense(1)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model Architecture][image8]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![320 x 160 x 3][image2]

Since my first trial is not success in the region of red line cornering, so a record and append more data on that region. 

After the collection process, I had 2155 number of data points. I then preprocessed this data by resize, crop and flip:
![320 x 160 x 3][image4]
center
![320 x 160 x 3][image5]
left with +0.2 steering angle correction
![320 x 160 x 3][image6]
right with -0.2 steering angle correction
![320 x 160 x 3][image7]
center Cropped
![320 x 160 x 3][image10]
left Cropped
![320 x 160 x 3][image11]
right Cropped
```
images = []
measurements = []

for line in lines:
    steering_center = float(line[3])
# create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

# read in images from center, left and right cameras
    path = "/home/carnd/CarND-Behavioral-Cloning-P3/data/IMG/"
    img_center = np.asarray(Image.open(path + line[0].split('\\')[-1]))
    img_left = np.asarray(Image.open(path + line[1].split('\\')[-1]))
    img_right = np.asarray(Image.open(path + line[2].split('\\')[-1]))
    img_center = cv2.resize(img_center, (80, 40))
    img_left = cv2.resize(img_left, (80, 40))
    img_right = cv2.resize(img_right, (80, 40))
# add images and angles to data set
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
```
To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:
![alt text][image3]
```
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
```

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by:
```
model.fit(X_train,y_train,validation_split=0.2, shuffle=True,nb_epoch=20)

```
I used an adam optimizer so that manually training the learning rate wasn't necessary. I didn't use any generators since my data was compressed as very small image, the training process goes very quick.
