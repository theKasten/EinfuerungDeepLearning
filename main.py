import numpy as np
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

x_one = np.array([-1, 0])
x_two = np.array([1, 0])
x_three = np.array([1, 1])
x = np.array([x_one, x_two])
y_one = 0
y_two = 1
y_three = 1
y = np.array([y_one, y_two])
w = np.array([-1, 0])
r = 1

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
print("----Now using Training Function----")
w = np.array([-1, 0])
p2 = Perzeptron(w, r)
p2.train(x, y)
