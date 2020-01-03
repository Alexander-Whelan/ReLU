import numpy as np

# TODO 1: implement the backward pass of Relu
class Relu():
    """
    Relu activation.

    The forward pass receives the input data (array) and exchanges any negative
    entry for zero.

    The backward pass should calculate the gradient of the maximum function in
    the forward pass and return it
    """
    def forward(self, x):
        self._hidden = np.maximum(0, x)
        return self._hidden

    def backward(self, x):
        grad = (x > 0) * 1.0
        return grad


# TODO 2: implement the update equation of a gradient descent optimizer
class GradientDescent():
    """
    Simple gradient-descent optimizer.
    """
    def step(self, weights, grad, step_size):
        weights = weights - (step_size * grad)
        return weights


if __name__ == "__main__":

    # Let's run some tests with the new functions.
    iterations = 1
    step_size = 0.01
    # Initialize weights
    weights = 0.01 * np.ones((20, 100))
    # Initialize artificial gradients
    grads = [0.5/(i+1) * np.ones((20, 100)) for i in range(iterations)]
    grads = np.array(grads)

    # Instantiate optimizer and activation
    optimizer = GradientDescent()
    relu = Relu()

    for i in range(iterations):
        print('weights = ' + str(weights))
        relu.forward(weights)
        print('weights = ' + str(weights))
        grad = relu.backward(weights)
        weights = optimizer.step(weights, grad, step_size)
        print('weights = ' + str(weights))
        print('grad =' + str(grad))
