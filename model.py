# imports
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Conv2D, Lambda

# extract csv data
print("Loading data...")
lines = []
with open("data/driving_log.csv") as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
print("Data stored.")

# Split csv data into training and validation
train_data, validation_data = train_test_split(lines, test_size=0.2)

def data_generator(samples, batch_size=32):
    """
    Data generator for better performance
    INPUT: samples = csv lines as list
    OUTPUT: (yielded) shuffeld augmented data as numpy array
    """
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch = samples[offset:offset+batch_size]
            
            images = []
            labels = []

            for line in batch:
                source = line[0]
                filename = source.split("/")[-1]
                path = "data/IMG/"
                image = cv2.imread(path + filename)
                measurement = float(line[3])
                images.append(image)
                labels.append(measurement)

                # Data Augmentation
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                labels.append(-measurement)

                # Left and Right camera
                correction = 0.2
                filename_l = line[1].split("/")[-1]
                filename_r = line[2].split("/")[-1]
                left_image = cv2.imread(path + filename_l)
                right_image = cv2.imread(path + filename_r)
                images.append(left_image)
                labels.append(measurement + correction)
                images.append(right_image)
                labels.append(measurement - correction)
          
            X_train = np.array(images)
            Y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, Y_train)

print("Training model...")

# Batch generation
batch_size = 32
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(validation_data, batch_size)

## Model
# Nvidea's end-to-end CNN (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
model = Sequential()
# Normalization and cropping
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))           # Normalization
model.add(Cropping2D(cropping=((70,25),(0,0))))
# 5 Convolutional layers
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))   
model.add(Conv2D(64, (3,3), activation="relu"))  
# 4 Fully Connected Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
history = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_data)/batch_size), validation_data=valid_generator, validation_steps=np.ceil(len(validation_data)/batch_size), epochs=10, verbose=1)
model.save("model.h5")
print("Model trained and saved as 'model.h5'.")

# Visualize loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
