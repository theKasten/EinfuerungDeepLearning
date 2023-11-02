import numpy as np
import matplotlib.pyplot as plt
import sklearn

from perzeptron import Perzeptron
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import time

import torch
from my_torch_model import Perzeptron as my_torch_perzeptron

class Perzeptron(torch.nn.Module):
    global weight
    def fit(self, xs, ys, epochs, with_bias, mute):
        self.mute = mute
        self.with_bias = with_bias
        if with_bias:
            xs = self.to_affine(xs)

        self.perceptron_fit_simd_minibatch(xs, ys, batchsize=epochs)

    def perceptron_fit_simd_minibatch(self, xs, ys, learn_rate=0.01, batchsize=1):
        self.weight = torch.ones_like(xs[0]) #torch.float64??
        accumulate_weight = self.weight
        for batch in range(len(xs)//batchsize):
            xs_batch = xs[batch*batchsize:(batch+1)*batchsize]
            ys_batch = ys[batch*batchsize:(batch+1)*batchsize]
            predictions = self.perceptron_predict_simd2(accumulate_weight, xs_batch)
            y_pred_vector = ys_batch - predictions
            aenderungs_vektoren = torch.ones_like(xs_batch)
            for row in range(xs_batch.shape[0]):
                aenderungs_vektoren[row, :] = y_pred_vector[row] * xs_batch[row, :]
            summe_aenderungen_in_x = torch.sum(aenderungs_vektoren, axis=0)
            self.weight += summe_aenderungen_in_x * learn_rate
            #self.show_information("Trained a batch!")

    def to_affine(self, x):
        #import numpy as np
        #print("This Matrix: ")
        #print(np.append(np.ones_like(x[:,0]).reshape(-1,1), x.numpy(), axis=1))
        #print("Should look like: ")
        affine_tensor = torch.cat([torch.ones_like(x[:,0]).reshape(-1,1), x], 1)
        #print(affine_tensor)
        return affine_tensor

    def perceptron_predict_simd2(self, weight_vector, xs):
        return torch.where(torch.tensordot(xs, weight_vector, 1) > 0, 1, 0)#Removed ", axes=1" when converting to torch model

    def get_decision_boundary(self):
        if(self.with_bias):
            bias, w = - self.weight[0], self.weight[1:]
            slope = - w[0] / w[1]
            intercept = bias / w[1]
        else:
            slope = - self.weight[0] / self.weight[1]
            intercept = 0
        return slope, intercept
def prep_perzeptron():
    plt.rcParams['figure.figsize'] = (4, 4)

    # Example data for binary classification:
    xs = np.array([[1,0.3], [-1,1.1], [0,-1.1], [0.8, -0.9]], dtype='float64')
    ys = np.array([0,       0,         0,        1], dtype='float64')

    # Let's have a look at the data:
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(xs[:,0], xs[:,1], c=ys)
    plt.show()

    xs_aff = Perzeptron.to_affine(xs)
    iterations = 0
    loss = 1
    #w = np.ones_like(xs_aff[0], dtype="float64")
    my_perzeptron = Perzeptron()
    my_perzeptron.train(xs_aff, ys)
    w = my_perzeptron.weight

    # Visualize Training
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = (-2, 2), ylim = (-2, 2))
    ax.scatter(xs[:,0], xs[:,1], c=ys)

    # compute decision boundary line:
    bias, weight = -w[0], w[1:]
    slope = - weight[0]/weight[1]
    intercept = bias / weight[1]

    ax.plot(boundary_x := np.linspace(-2, 2, 2), slope * boundary_x + intercept)
    ax.arrow(bias/weight[0], 0, weight[0], weight[1],
             width=.03, length_includes_head=True)

    plt.show()

def test_minibatch():
    iterations = 0
    loss = 1

    my_perzeptron = Perzeptron()

    # Example data for binary classification:
    xs = np.array([[1,0.3], [-1,1.1], [0,-1.1], [0.8, -0.9]])
    ys = np.array([  0,      0,       0,          1])

    #my_perzeptron.fit(xs, ys, 4, True, False)
    #plot_prediction_with_boundary(xs, ys, epochs=1, with_bias=False)#Mit Bias schlägt multiplikaion in batch prozessing fehl!
    #plot_prediction_with_boundary(xs, ys, epochs=2, with_bias=False)#Mit Bias schlägt multiplikaion in batch prozessing fehl!
    #plot_prediction_with_boundary(xs, ys, epochs=3, with_bias=False)#Mit Bias schlägt multiplikaion in batch prozessing fehl!
    plot_prediction_with_boundary(xs, ys, epochs=4, with_bias=False)#Mit Bias schlägt multiplikaion in batch prozessing fehl!

def plot_prediction_with_boundary(xs, ys, epochs=200, with_bias=True):
    P = Perzeptron()
    P.fit(xs, ys, epochs=epochs, with_bias=with_bias, mute=True)
    slope, intercept = P.get_decision_boundary()
    #print("current loss:", Perceptron.loss(ys, P.predict(xs)))

    #compute ideal square plotting area:
    xmin, xmax = np.min(xs[:,0]), np.max(xs[:,0])
    ymin, ymax = np.min(xs[:,1]), np.max(xs[:,1])
    smallest = np.min([xmin, ymin]) - 1
    biggest = np.max([np.abs(xmax-xmin), np.abs(ymax-ymin)]) + 2
    xlim = (smallest, smallest + biggest)
    ylim = (smallest, smallest + biggest)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = xlim, ylim = ylim)
    ax.scatter(xs[:,0], xs[:,1], c=ys)
    ax.plot(boundary_x := np.linspace(smallest, biggest, 2),
            slope * boundary_x + intercept)
    plt.show()

def evaluation_auf_Iris():
    xs, ys = load_iris(return_X_y=True)
    xs, ys = xs[ys != 2], ys[ys != 2]
    xs = PCA(2).fit_transform(xs)


    start = time.time()
    plot_prediction_with_boundary(xs, ys, epochs=1)
    end = time.time()
    print("Batch size = 1 took " + str(end - start))

    start = time.time()
    plot_prediction_with_boundary(xs, ys, epochs=4)
    end = time.time()
    print("Batch size = 4 took " + str(end - start))

def py_torch_und_iris():
    if torch.cuda.is_available():
        print("Cuda available, now strating...")
    xs, ys = load_iris(return_X_y=True)
    xs, ys = xs[ys != 2], ys[ys != 2]
    xs = PCA(2).fit_transform(xs)

    my_torch_p = my_torch_perzeptron()
    my_torch_p.fit(torch.from_numpy(xs), torch.from_numpy(ys), epochs=4, with_bias=True, mute=True)
    slope, intercept = my_torch_p.get_decision_boundary()
    #print("current loss:", Perceptron.loss(ys, P.predict(xs)))

    #compute ideal square plotting area:
    xmin, xmax = np.min(xs[:,0]), np.max(xs[:,0])
    ymin, ymax = np.min(xs[:,1]), np.max(xs[:,1])
    smallest = np.min([xmin, ymin]) - 1
    biggest = np.max([np.abs(xmax-xmin), np.abs(ymax-ymin)]) + 2
    xlim = (smallest, smallest + biggest)
    ylim = (smallest, smallest + biggest)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = xlim, ylim = ylim)
    ax.scatter(xs[:,0], xs[:,1], c=ys)
    ax.plot(boundary_x := np.linspace(smallest, biggest, 2),
            slope * boundary_x + intercept)
    plt.show()

def py_torch_und_moon():
    X, y = sklearn.datasets.make_moons(n_samples=500, noise=0.3, random_state=40)
    X = PCA(2).fit_transform(X)

    my_torch_p = my_torch_perzeptron()
    my_torch_p.fit(torch.from_numpy(X), torch.from_numpy(y), epochs=4, with_bias=True, mute=True)
    slope, intercept = my_torch_p.get_decision_boundary()
    #print("current loss:", Perceptron.loss(ys, P.predict(xs)))

    #compute ideal square plotting area:
    xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
    ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
    smallest = np.min([xmin, ymin]) - 1
    biggest = np.max([np.abs(xmax-xmin), np.abs(ymax-ymin)]) + 2
    xlim = (smallest, smallest + biggest)
    ylim = (smallest, smallest + biggest)

    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = xlim, ylim = ylim)
    ax.scatter(X[:,0], X[:,1], c=y)
    ax.plot(boundary_x := np.linspace(smallest, biggest, 2),
            slope * boundary_x + intercept)
    plt.show()

#prep_perzeptron()
#test_minibatch()
#evaluation_auf_Iris()
#py_torch_und_iris()
py_torch_und_moon()

