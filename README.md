# Neural Network with Batch Normalization, Dropout and Different Optimizers


This repository contains the implementation of a simple neural network model with batch normalization and dropout layers for regularizing and improving the training of the model. Furhtermore they are then tested using with and without the specified implemenation above using two different optimizers which are ADAM and SGD. The model was written using the pytorch library.

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

Experiments show that Adam outperformed SGD by approximately 4% in average when comparing 15 trained models from each optimzer. This superiority can be distilled into:

- Adaptive Mechanism: Adam's ability to tweak learning rates based on past gradients aids in navigating the loss surface more efficiently.
- Initial Choices: SGD's performance is more susceptible to initial parameter values and learning rate choices.
- Dataset Discrepancies: Some datasets or problems inherently lean towards one optimizer.

## Recommendations for future Tests:

- Hyperparameter Mastery: Always fine-tune hyperparameters for each optimizer. Default values aren't universally optimal.
- Regularization Dynamics: If using regularization, remember that its efficacy can vary between optimizers. Tune accordingly.
- Employ Learning Rate Scheduling: Strategies like learning rate decay can markedly amplify the performance of optimizers, especially SGD.
- Data Augmentation: Adding random horizontal/vertical flips and zooms can help both optimizers to get better accuracy on the test data.
- Altough we utilized PyTorch for the model development, it's worth noting that using the Keras Tuner could further enhance our outcomes. By automating the exploration of various optimizers and learning rates.

  
# Adam vs. SGD: A Pictorial Analysis

### ADAM

![figure1adam](https://github.com/kamileren/MNISTOptim/blob/main/images/Adam.png
)

figure 1 (Adam with out Batch Normilization)


![figure2adam](https://github.com/kamileren/MNISTOptim/blob/main/images/Adam2.png
)


figure2 (Adam with batch normalization and dropout)

![figure3adam](https://github.com/kamileren/MNISTOptim/blob/main/images/Adam3.png
)

figure3 (Adam with dropout)


### SGD

![figure1sgd](https://github.com/kamileren/MNISTOptim/blob/main/images/SGD.png
)

figure1 (SGD without batch normalization)

![figure2sgd](https://github.com/kamileren/MNISTOptim/blob/main/images/SGD2.png
)

figure2 (SGD without batch normalization and dropout)

![figure3sgd](https://github.com/kamileren/MNISTOptim/blob/main/images/SGD3.png
)

figure3 (SGD with batch normalization and dropout)

Batch normalization and dropout stand out as transformative techniques in neural network optimization. Batch normalization ensures consistent learning by minimizing internal shifts in activation distributions, thus paving the way for faster convergence. It also acts as a mild regularizer by introducing minor noise during training. On the other hand, dropout enhances a model's generalization. It deactivates random neurons during training, preventing them from becoming overly specialized and making the network more resilient through enforced redundancy. When combined, these techniques offer a powerful tandem. While batch normalization provides stable and swift learning, dropout ensures the model doesn't lean too heavily on specific neurons. In the context of your model trained with SGD, the integration of these techniques likely addressed challenges related to both slow convergence and overfitting, leading to a significant boost in accuracy.

## Testing the Final Model (NNP.pth) (NEW MODEL)

Utilizing our optimal model, which was trained with SGD complemented by batch normalization and dropout, we achieved a remarkable accuracy of **99.18%**. This stands in stark contrast to the average accuracy attained using Adam, which was 97%. Thus, there's a substantial improvement of 2% in accuracy. Even more impressively, when compared with the average performance of SGD at 93%, we observe a significant leap of 6%. 




![tests](https://github.com/kamileren/MNISTOptim/blob/main/images/Tests.png
)


As demonstrated above, the model adeptly discerned even the most unrecognizable numerical representations in the test set.

## Conclusion

Throughout this analysis, we've ventured deep into the intricacies of various optimization techniques and their impacts on model performance. Our findings underscore the transformative power of combining strategic regularization techniques with the right optimizer. Notably, our model's ability to adeptly discern even the most atypical numerical representations is testament to its robustness and efficacy.





