import numpy as np
import sklearn as sk
from random import randint

from sklearn.model_selection import train_test_split

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
    print('---Now Generating Data...---')
    generated_data_matrix = Perzeptron.generate(10, 10)
    print(generated_data_matrix)
    generated_x = generated_data_matrix[:10, :];
    generated_y = generated_data_matrix[10, :]
    print('X: ' + str(generated_x))
    print('Y: ' + str(generated_y))

    print('---Starting Test Train Split---')
    X_train, X_test, y_train, y_test = train_test_split(generated_x, generated_y, test_size=0.33, random_state=42)
    print('X_train: ' + str(X_train))
    print('X_test: ' + str(X_test))
    print('y_train: ' + str(y_train))
    print('y_test: ' + str(X_test))

aufgabe_02_03() 