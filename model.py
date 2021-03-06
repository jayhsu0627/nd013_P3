import csv
import cv2
import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array, load_img
lines = []
with open('/home/carnd/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
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
            
print(measurements)
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x:((x/255.0)-0.5),input_shape=(40,80,3)))
## Cropping Dimension influence the Con2D parameters
model.add(Cropping2D(cropping=((17,6),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Flatten())
model.add(Dense(60))
model.add(Dense(42))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2, shuffle=True,nb_epoch=20)

from keras.models import Model


model.save('model.h5')
