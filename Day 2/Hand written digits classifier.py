import numpy as np
import random as rd
import matplotlib.pyplot as plt
from keras import layers, models, datasets
from keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels) , (test_images, test_labels) = datasets.mnist.load_data()

# Preprocess the data : normalize the pixel to be between 0 and 1
train_images = train_images/225.0
test_images = test_images/225.0

# Reshape the images to (28, 28, 1) as they are grayscale images
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Convert the labels to one hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the CNN model
model = models.Sequential()

# First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the 3D output to 1D and add a dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 neurons for 10 digit classes
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model on test data
test_loss , test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100 :.2f}%")

TEST_CASES = 10
for j in range(TEST_CASES):
    
    # Make prediction for that image
    i = rd.randint(0, len(test_images)-1)
    prediction = model.predict(test_images[i:i+1], verbose=0)
    predicted_label = np.argmax(prediction)

    print(f"Prediction for test image {i}: {predicted_label}")

    # Show the image
    plt.figure(num=f"Figure {j+1}", figsize=(4, 4))
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis("off")
    plt.show()

