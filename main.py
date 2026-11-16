import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

data = pd.read_csv("./data/train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T  # 785 * 1000
X_dev = data_dev[1:n] / 255  # 784 * 1000
Y_dev = data_dev[0]  # 1 * 1000

data_train = data[1000:m].T  # n * m'
X_train = data_train[1:n] / 255  # 784 * m'
Y_train = data_train[0]  # 1 * m'


def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # 10 * 784
    b1 = np.random.rand(10, 1) - 0.5  # 10 * 1
    W2 = np.random.rand(10, 10) - 0.5  # 10 * 10
    b2 = np.random.rand(10, 1) - 0.5  # 10 * 1
    return W1, b1, W2, b2


def relu(i):
    return np.maximum(0, i)


def der_relu(i):
    return i > 0


def softmax(i):
    exp_shifted = np.exp(i - np.max(i, axis=0, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=0)


def fwd_prop(W1, b1, W2, b2, x):
    z1 = W1.dot(x) + b1  # 10 * m'
    a1 = relu(z1)  # 10 * m'
    z2 = W2.dot(a1) + b2  # 10 * m'
    a2 = softmax(z2)  # 10 * m'
    return z1, a1, z2, a2


def one_hot(Y):  # 1 * m'
    one_hot_y = np.zeros((Y.size, Y.max() + 1))  # m' * 10
    one_hot_y[np.arange(Y.size), Y] = (
        1  # looks at each row and adds a 1 when its at the position of the number stored in Y
    )
    one_hot_y = one_hot_y.T  # 10 * m'
    return one_hot_y  # 10 * m'


def back_prop(z1, a1, z2, a2, w1, w2, x, y):
    m = y.size
    one_hot_y = one_hot(y)  # 10 * m'
    dZ2 = a2 - one_hot_y  # 10 * m'
    dW2 = 1 / m * dZ2.dot(a1.T)  # 10 * 10
    dZ1 = w2.T.dot(dZ2) * der_relu(z1)
    dW1 = 1 / m * dZ1.dot(x.T)  # 10 * 10
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def updateParams(dW1, dW2, db1, db2, W1, W2, b1, b2, alpha):
    W1 -= dW1 * alpha
    W2 -= dW2 * alpha
    b1 -= db1 * alpha
    b2 -= db2 * alpha
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def grad_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = fwd_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParams(dW1, dW2, db1, db2, W1, W2, b1, b2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = grad_descent(X_train, Y_train, m - 2000, 0.1)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = fwd_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()


print("\n--- Visualizing Predictions ---")
for i in range(10):
    # Pick a random image index anywhere in our training dataset size (m')
    random_index = random.randint(0, X_train.shape[1] - 1)

    print(f"\n--- Test Sample {i + 1} (Dataset Index: {random_index}) ---")
    test_prediction(random_index, W1, b1, W2, b2)
