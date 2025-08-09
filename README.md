# ANN---Artificial-Nural-Network

🔹 Step 1: ANN-1.ipynb Explanation
This notebook contains a basic Artificial Neural Network (ANN) built using TensorFlow/Keras for a classification task on the Fashion MNIST dataset.

Key Code Highlights:
Data Loading:

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
Loads 60,000 training and 10,000 test grayscale images (28x28) of fashion items.

Normalization:

x_train = x_train / 255
x_test = x_test / 255
Pixel values scaled between 0 and 1 for faster convergence.

Model Definition:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
Flatten Layer: Converts 28x28 image to a 784-length vector.
Dense Layer 1: 128 neurons with ReLU activation.
Output Layer: 10 neurons with softmax for multi-class classification (10 fashion categories).

Compilation:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Uses Adam optimizer and categorical crossentropy for multiclass classification.

Training:

model.fit(x_train, y_train, epochs=5)
Trains the model for 5 epochs.

Evaluation:

model.evaluate(x_test, y_test)

Tests model on unseen data and prints accuracy.

