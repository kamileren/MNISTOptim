# Neural Network with Batch Normalization, Dropout and Different Optimizers


This repository contains the implementation of a simple neural network model with batch normalization and dropout layers for regularizing and improving the training of the model. Furhtermore they are then tested using with and without the specified implemenation above using two different optimizers which are ADAM and SGD.

## Network Architecture

The neural network begins with a fully connected layer that takes input vectors of size **28 * 28** (which is $28^2$), transforming them into an intermediate representation of 100 units. This transformation is followed by a batch normalization layer which is designed to stabilize and accelerate the learning process. This layer normalizes the output of the previous fully connected layer for the 100 units.

Subsequent to the normalization, the ReLU (Rectified Linear Unit) activation function is applied, introducing non-linearity into the model. This non-linearity allows the network to learn more complex relationships in the data.

To prevent the model from overfitting and to introduce some form of regularization, a dropout layer is added next. This layer randomly sets 50% of the input units to 0 during training, which helps the model to be more robust and reduces its reliance on any specific neuron.

The network then transitions to another fully connected layer which reduces the 100 units from the previous layers to 50 units. Just like before, this is followed by another batch normalization layer, specifically tailored for these 50 units. After normalizing, the ReLU activation function is applied again to introduce non-linearity.

Another dropout layer with a rate of 0.5 is then added for the same regularization purposes as mentioned earlier.

Finally, the network concludes with a fully connected layer that maps the 50 units to 10 output units. These 10 units can be thought of as the final classification scores or logits for a 10-class problem, for instance.

This architecture effectively combines linear transformations (through the fully connected layers) with non-linearities (through ReLU) and regularization techniques (batch normalization and dropout) to achieve a robust and versatile model.


# Optimizer Showdown: Adam vs. SGD

From now on we will be talkning about the difference between the two used optimizers Adam and Stochastic Gradient Descent (SGD). Based on extensive experiments and observations, here's a  breakdown of their intricacies, strengths, and weaknesses on the MNIST dataset.

### Stochastic Gradient Descent (SGD):

- Straightforward Update: Updates parameters based on the gradient concerning a single training instance.
- Learning Rate Nuances: The learning rate needs fine-tuning. An excessively high rate can diverge the model, while a low one can lead to painfully slow training.
- Zig-Zag Convergence: The convergence pattern can be slow, especially in regions known as saddle points.
- Uniform Updates: Every parameter update is consistent, implying all parameters get the same learning rate.
- Momentum Boost: Variants with momentum can address some inherent challenges, taking into account past gradients.


### Adam:

- Smart Learning Rates: Adam dynamically adjusts learning rates for each parameter, leveraging first and second moments of the gradients.
- Complex Initialization: Requires moving averages and a bias-correction term initialization, which naturally complicates the update rule.
- Rapid Convergence: Typically outpaces SGD, especially for noisy gradients or sparse datasets.
- Learning Rate Flexibility: Generally more forgiving when it comes to initial learning rate selection.
- Memory Considerations: Consumes more memory than SGD since it maintains per-parameter moving averages.



## Experimental Insights:

Experiments show that Adam outperformed SGD by approximately 5% in average when comparing 15 trained models from each optimzer. This superiority can be distilled into:

- Adaptive Mechanism: Adam's ability to tweak learning rates based on past gradients aids in navigating the loss surface more efficiently.
- Initial Choices: SGD's performance is more susceptible to initial parameter values and learning rate choices.
- Dataset Discrepancies: Some datasets or problems inherently lean towards one optimizer.

## Recommendations for future Tests:

- Hyperparameter Mastery: Always fine-tune hyperparameters for each optimizer. Default values aren't universally optimal.
- Regularization Dynamics: If using regularization, remember that its efficacy can vary between optimizers. Tune accordingly.
- Employ Learning Rate Scheduling: Strategies like learning rate decay can markedly amplify the performance of optimizers, especially SGD.
- Data Augmentation: Adding random horizontal/vertical flips and zooms can help both optimizers to get better accuracy on the test data.

  






