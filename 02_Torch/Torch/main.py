import numpy as np
import matplotlib.pyplot as plt
from perzeptron import Perzeptron
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

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
    xs = np.array([[1,0.3], [-1,1.1], [0,-1.1], [0.8, -0.9]], dtype='float64')
    ys = np.array([         0,      0,       0,          1])

    while(loss > 0.1 and iterations < 500):
        my_perzeptron.perceptron_fit_simd_minibatch(xs, ys,
                                          learn_rate=0.01,
                                          batchsize=1)
        y_pred = my_perzeptron.activating_function(xs)
        loss = np.sum(y_pred != ys) / len(ys)
        if(iterations % 50 == 0): # only every 50th iteration:
            print(iterations, "Gewichtsvektor: ", my_perzeptron.w)
            print("Anteil falsch klassifiziert: ", loss)
        iterations += 1

        print("actual labels:    ", ys)
        print("predicted labels: ", y_pred)

def evaluation_auf_Iris:
    xs, ys = load_iris(return_X_y=True)
    xs, ys = xs[ys != 2], ys[ys != 2]
    xs = PCA(2).fit_transform(xs)

    plot_prediction_with_boundary(xs, ys, epochs=4)



#prep_perzeptron()
#test_minibatch()
evaluation_auf_Iris()