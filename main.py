import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn import datasets
from perzeptron import Perzeptron

#print("Hello World!")
#a = np.arange(15).reshape(3, 5)

#print("Value of a: ")
#print(a)

#print(p.f())

"""
x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
y -> Lable
w(z)? -> Gewicht??
r -> lernrate
"""

def aufgabe_01():
    x_one = np.array([-1, 0])
    x_two = np.array([1, 0])
    x_three = np.array([1, 1])
    x = np.array([x_one, x_two])
    y_one = 0
    y_two = 1
    y_three = 1
    y = np.array([y_one, y_two])
    w = np.array([-1, 0])
    r = 0.05

    p = Perzeptron(w, r)

    #Predict
    prediction = p.activating_function(x_three)
    print('Before Training Prediction for x three was class:' + str(prediction))

    #Train
    w = p.weight_new(x_one, y_one)
    print('New w is now:' + str(w))
    w = p.weight_new(x_two, y_two)
    print('New w is now:' + str(w))


    #Predict
    prediction = p.activating_function(x_three)
    print('Prediction for x three was class:' + str(prediction))

    print()

    x_test = np.array([[100, 5, -2],
                       [100, -1, 30]])

    print("----Now using Training Function----")
    w = np.array([-1, 0])
    p2 = Perzeptron(w, r)
    p2.train(x, y)
    print('Lables: ' + str(p2.infer(x_test)))

    print()
    print("---Now using Testing for training---")
    p3 = Perzeptron(np.array([randint(-10, 10), randint(-10, 10)]), r)#Mit zufälligem w
    p3.train(x_test, np.array([1, 1, 0]))
    print('Lables: ' + str(p3.infer(x_test)))

def aufgabe_02_03():
    number_of_features = 3;
    dimension = 10000;

    print('---Now Generating Data...---')
    generated_data_matrix = Perzeptron.generate(number_of_features, dimension)
    #print(generated_data_matrix)
    generated_x = generated_data_matrix[:number_of_features, :]
    generated_y = generated_data_matrix[number_of_features, :]
    #print('X: ' + str(generated_x) + '\nX.shape: ' + str(generated_x.shape))
    #print('Y: ' + str(generated_y) + '\nY.shape: ' + str(generated_y.shape))

    print('---Starting Test Train Split---')
    X_train, X_test, y_train, y_test = train_test_split(np.transpose(generated_x), generated_y, test_size=0.33, random_state=42)
    #print('X_train: ' + str(X_train))
    #print('X_test: ' + str(X_test))
    #print('y_train: ' + str(y_train))
    #print('y_test: ' + str(X_test))

    print('---Starting Training---')
    w_init = np.zeros(number_of_features)
    for n in range(number_of_features):
        w_init[n] = np.array([randint(-100, 100)])
    my_freshly_trained_p = Perzeptron(w_init, 0.01)
    my_freshly_trained_p.train(np.transpose(X_train), y_train)

    print('---Starting evaluation---')
    y_predictions = my_freshly_trained_p.infer(np.transpose(X_test))
    #print("y_test data: " + str(y_test))
    #print("y_predictions: " + str(y_predictions))
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    print('---Finished evaluation---')
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy: ' + str(accuracy) + '%')
    print('Error Prob: ' + str(100 - accuracy) + '%')

def aufgabe_04():
    print('---Evaluation mit Iris-Flower-Datasets---')
    iris = datasets.load_iris()
    #for keys in iris.keys() :
    #    print(keys)
    split = 0;
    for i in iris.target:
        if iris.target[i] <= 1:
            split = split + 1

    X=iris.data[:100, :]
    y=iris.target[:100]#Ab hier fangen die anderen blumen an

    #print(y)
    print("---Starte Test Train split---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #print('X_train: ' + str(X_train))
    #print('X_test: ' + str(X_test))
    print('y_train: ' + str(y_train))
    #print('y_test: ' + str(X_test))
    print("---Starte Training---")
    number_of_features = X_train.shape[1]
    #w_init = np.zeros(number_of_features)
    w_init = np.random.uniform(-1, 1, number_of_features)
    """for n in range(number_of_features):
        w_init[n] = np.array([randint(-10, 10)])#-> Randmoisierung von initialem w auch ausschlaggebend, wie viel durch wenige daten angepasst werden kann
    """
    my_fresh_p = Perzeptron(w_init, 0.01)#-> Muss deutlich höher geschraubt werden als 0.01, da nicht so viele Train Daten vorhanden sind.
    my_fresh_p.train(np.transpose(X_train), y_train)
    my_fresh_p.show_information()

    print('---Starting evaluation---')
    y_predictions = my_fresh_p.infer(np.transpose(X_test))
    print("y_test data: " + str(y_test))
    print("y_predictions: " + str(y_predictions))
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    print('---Finished evaluation---')
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy: ' + str(accuracy) + '%')
    print('Error Prob: ' + str(100 - accuracy) + '%')

#aufgabe_01()
#aufgabe_02_03()
aufgabe_04()