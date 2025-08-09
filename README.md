# ANN---Artificial-Nural-Network
**🔹  ANN-1.ipynb**

-> This file builds a basic artificial neural network (ANN) to classify images from the Fashion MNIST dataset, which contains grayscale images of clothing items (like shirts, shoes, etc.). The key steps are:

-> It loads the dataset and prepares it by normalizing the pixel values.

-> The model has three layers: an input layer to flatten the image, a hidden layer to learn patterns, and an output layer to predict one of the 10 clothing categories.

-> The model is trained on the training data for a few cycles (epochs).

-> Finally, the model is tested on new images to check how accurately it predicts.


**🔹ANN-2.ipynb**


-> This file builds upon the previous model and includes performance analysis and visualization:

-> After training the model, it uses prediction results to analyze how well the model performed.

-> It calculates accuracy and also uses tools like confusion matrix and classification report to give a detailed view of where the model is getting confused.

-> It shows which images were predicted incorrectly by the model and compares the wrong predictions with the actual correct labels.

-> This helps understand the model’s strengths and weaknesses more clearly.


**🔹 ANN-3.ipynb **
-> This file goes a step further by introducing model tuning and data splitting:

-> Instead of using the default data split, it manually splits the dataset into training and testing parts.

-> It adds dropout, a regularization technique to prevent the model from overfitting (memorizing data too closely).

-> The model structure may be slightly adjusted (e.g., more neurons or layers), and the training is done with validation checks to monitor performance.

-> This version shows a more professional and controlled way of training neural networks for better generalization.
