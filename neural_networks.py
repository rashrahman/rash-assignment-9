import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function
        self.activations = {'tanh': np.tanh, 
                            'relu': lambda x: np.maximum(0, x), 
                            'sigmoid': lambda x: 1 / (1 + np.exp(-x))}
        self.activation_derivatives = {'tanh': lambda x: 1 - np.tanh(x) ** 2,
                                       'relu': lambda x: (x > 0).astype(float),
                                       'sigmoid': lambda x: self.activations['sigmoid'](x) * (1 - self.activations['sigmoid'](x))}
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # Forward pass
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activations[self.activation_fn](self.Z1)  # Activation
        self.Z2 = self.A1 @ self.W2 + self.b2
        out = self.activations['sigmoid'](self.Z2)  # Output activation (sigmoid for binary classification)
        self.out = out
        return out

    def backward(self, X, y):
        # Compute gradients using chain rule
        m = X.shape[0]
        dZ2 = self.out - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivatives[self.activation_fn](self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Layer Features")

    # Decision boundary in input space
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap="bwr", alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title("Input Space Decision Boundary")

    # Gradient visualization
    gradient_magnitude = np.sqrt(np.sum(mlp.W1 ** 2, axis=1))
    for i, mag in enumerate(gradient_magnitude):
        circle = Circle((mlp.W1[i, 0], mlp.W1[i, 1]), mag, alpha=0.5)
        ax_gradient.add_patch(circle)
    ax_gradient.set_xlim(-1, 1)
    ax_gradient.set_ylim(-1, 1)
    ax_gradient.set_title("Gradients")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)