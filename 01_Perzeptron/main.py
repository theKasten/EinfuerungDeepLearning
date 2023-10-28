import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
from random import randint
import numpy as np

class Perzeptron:
    global w    #weight
    global r    #lernrate
    global b    #liasterm

    @staticmethod
    def to_affine(x):
        return np.append(np.ones_like(x[:, 0]).reshape(-1, 1), x, axis=1)

    def __init__(self, w_init, learn_rate):
        self.w = w_init
        self.r = learn_rate
        self.b = 0;
        #print('Perzeptron has been initialised with initial weight of ' + str(self.w)
        #      + ' and initial learn rate ' + str(self.r) + '...')

    def infer(self, x):
        """
        INPUT
        x -> Eingabe Vektor
        RETURN
        y -> Lables in das Perzeptron die Eingabevektoren laut seinen internen Gewichten stecken würde
        """
        y = np.zeros(x.shape[1], dtype=int)
        for k in range(x.shape[1]):
            y[k] = self.activating_function(x[:, k])

        return y

    def activating_function(self, x):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        w -> Gewich
        :return:
        Vorhersage der Binären klassifikation von x mit gewicht w
        (Also gehört x Klasse 0 oder 1 an)
        """
        if np.dot(self.w, x + self.b) > 0:
            #print("Element in klass 1")
            return 1
        else:
            #print("Element in klass 0")
            return 0

    def weight_new(self, x, y):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        y -> Lable
        w(z)? -> Gewicht??
        r -> lernrate
        :return:
        """
        self.w = (self.w + self.r * (y - self.activating_function(x)) * x)#TODO: So richtig??
        #w = w + np.dot(r * (y - self.activating_function(w, x)) , x)#TODO: So richtig??

    def train(self, x, y):
        #print("Shape x:" + str(x.shape))
        for k in range(x.shape[0]):#Schnappt sich Eizelne Zeilen aus der Matrix x. Die einzelnen Zeilen haben n Spalten/Features
            self.weight_new(x[k, :], y[k])#TODO: So richtig? Müssen hier Zeilen oder doch Spalten von X verwendet werden?
            #print('New w is now:' + str(self.w))

    @staticmethod
    def generate(number_of_features, dimension):
        """Soll synthetische Daten generieren!"""
        """Matrix mit n Feature vielen Zeilen + 1 Zeile für Lables, Dimension d/Anzahl der Spalten Heißt Anzahl der Eingabevektoren"""
        #w_init = np.array([randint(-100, 100), randint(-100, 100)])
        #Winit
        w_init = np.random.uniform(-1, 1, number_of_features)
        matrix = np.zeros((number_of_features + 1, dimension))# + 1 für lables
        p = Perzeptron(w_init, 0.001)

        for d in range(dimension):
            for n in range(number_of_features + 1):
                if n == number_of_features:
                    matrix[n, d] = p.activating_function(matrix[:number_of_features, d])
                else:
                    matrix[n, d] = randint(-100, 100)
        return matrix

    def show_information(self):
        print('Perzeptron has weight of ' + str(self.w)
              + ' and learn rate ' + str(self.r) + '...')

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
    p3.train(np.transpose(x_test), np.array([1, 1, 0]))
    print('Lables: ' + str(p3.infer(x_test)))

def aufgabe_02_03():
    number_of_features = 3;
    dimension = 10000;

    #Now Generating Data
    generated_data_matrix = Perzeptron.generate(number_of_features, dimension)
    generated_x = generated_data_matrix[:number_of_features, :]
    generated_y = generated_data_matrix[number_of_features, :]
    #Starting Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(np.transpose(generated_x), generated_y, test_size=0.33, random_state=42)
    #Init Perzeptron
    w_init = np.random.uniform(-1, 1, number_of_features)
    my_freshly_trained_p = Perzeptron(w_init, 0.01)
    my_freshly_trained_p.train(X_train, y_train)
    #Calc Predictions
    y_predictions = my_freshly_trained_p.infer(np.transpose(X_test))
    #Calc Accuracy
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    #Finished evaluation
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy Aufgabe 3: ' + str(accuracy) + '%')
def aufgabe_04():
    #Evaluation mit Iris-Flower-Datasets
    iris = datasets.load_iris()
    X=iris.data[:100, :]
    y=iris.target[:100]
    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    number_of_features = X_train.shape[1]
    #Perzeptron init
    w_init = np.random.uniform(-1, 1, number_of_features)
    my_fresh_p = Perzeptron(w_init, 0.01)#-> Muss deutlich höher geschraubt werden als 0.01, da nicht so viele Train Daten vorhanden sind.
    #Training
    my_fresh_p.train(X_train, y_train)
    #Predictions calc
    y_predictions = my_fresh_p.infer(np.transpose(X_test))
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    #Finished evaluation
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy Aufgabe 4: ' + str(accuracy) + '%')

def visualisierung_generator():
    number_of_features = 2;
    dimension = 1000;

    print('---Now Generating Data...---')
    generated_data_matrix = Perzeptron.generate(number_of_features, dimension)
    #print(generated_data_matrix)
    generated_x = generated_data_matrix[:number_of_features, :]
    generated_y = generated_data_matrix[number_of_features, :]

    X_train, X_test, y_train, y_test = train_test_split(np.transpose(generated_x), generated_y, test_size=0.33, random_state=42)
    number_of_features = X_train.shape[1]
    w_init = np.random.uniform(-1, 1, number_of_features)
    my_fresh_p = Perzeptron(w_init, 0.01)#-> Muss deutlich höher geschraubt werden als 0.01, da nicht so viele Train Daten vorhanden sind.
    my_fresh_p.train(X_train, y_train)

    print('---Visualisiere---')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = (-100, 100), ylim = (-100, 100))
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, marker='x')
    ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test, marker='*')

    #bias, weight = 0, my_fresh_p.w
    slope = my_fresh_p.w[1] / my_fresh_p.w[0]
    intercept = 0;
    print(my_fresh_p.w)
    ax.plot(boundary_x := np.linspace(-200, 200, 2), slope * boundary_x + intercept)

    y_predictions = my_fresh_p.infer(np.transpose(X_test))
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    print('---Finished evaluation---')
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy: ' + str(accuracy) + '%')

    plt.show()

def visualisierung_iris():
    iris = datasets.load_iris()

    X=iris.data[:100, :]
    y=iris.target[:100]#Ab hier fangen die anderen blumen a

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    number_of_features = X_train.shape[1]
    w_init = np.random.uniform(-1, 1, number_of_features)
    my_fresh_p = Perzeptron(w_init, 0.01)#-> Muss deutlich höher geschraubt werden als 0.01, da nicht so viele Train Daten vorhanden sind.
    my_fresh_p.train(X_train, y_train)

    print('---Visualisiere---')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = (-10, 10), ylim = (-10, 10))
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, marker='x')
    #ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test, marker='*')


    #bias, weight = 0, my_fresh_p.w[1:]
    print(my_fresh_p.w)
    slope = my_fresh_p.w[0] / my_fresh_p.w[1]
    #intercept = my_fresh_p.w[0] / my_fresh_p.w[1]
    intercept = 0

    ax.plot(boundary_x := np.linspace(0, 2, 2), slope * boundary_x + intercept)
    #ax.arrow(bias/weight[0], 0, weight[0], weight[1], width=.03, length_includes_head=True)


    plt.show()

    y_predictions = my_fresh_p.infer(np.transpose(X_test))
    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    print('---Finished evaluation---')
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy: ' + str(accuracy) + '%')

def visualisierung_mit_bias():
    print("---Bias einbauen---")
    number_of_features = 2;
    dimension = 100;

    print('---Now Generating Data...---')
    generated_data_matrix = Perzeptron.generate(number_of_features, dimension)
    #print(generated_data_matrix)
    generated_x = generated_data_matrix[:number_of_features, :]
    generated_y = generated_data_matrix[number_of_features, :]

    X_train, X_test, y_train, y_test = train_test_split(np.transpose(generated_x), generated_y, test_size=0.33, random_state=42)

    number_of_features = X_train.shape[1] + 1##affine + 1
    w_init = np.random.uniform(-1, 1, number_of_features)
    my_fresh_p = Perzeptron(w_init, 0.01)

    affine_x_train = my_fresh_p.to_affine(X_train);
    #print(affine_x_train)
    my_fresh_p.train(affine_x_train, y_train)


    print('---Visualisiere---')
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(xlim = (-100, 100), ylim = (-100, 100))
    ax.scatter(affine_x_train[:, 1], affine_x_train[:, 2], c = y_train, marker='x')
    #print(y_test)
    ax.scatter(Perzeptron.to_affine(X_test)[:, 1], Perzeptron.to_affine(X_test)[:, 2], c = y_test, marker='*')

    #ax.plot(boundary_x := np.linspace(0, 2, 2), 0 * boundary_x + 0)

    ax.arrow(0, 0, my_fresh_p.w[1]*20, my_fresh_p.w[2]*20, width=2, length_includes_head=False)
    slope = - my_fresh_p.w[1] / my_fresh_p.w[2]
    ax.plot(boundary_x := np.linspace(-100, 100, 2), slope * boundary_x + 0)


    #print(my_fresh_p.w)



    #ax.arrow(bias/weight[0], 0, weight[0], weight[1], width=.03, length_includes_head=True)

    #fig2, ax2 = plt.subplots()
    #ax2.grid()
    #ax2.set(xlim = (-1, 8), ylim = (-1, 8))
    y_predictions = my_fresh_p.infer(np.transpose(Perzeptron.to_affine(X_test)))
    print(y_predictions)
    affine_x_test = my_fresh_p.to_affine(X_test);
    #ax2.scatter(affine_x_test[:, 1], affine_x_test[:, 2], c = y_predictions, marker='x')

    bias, weight = -my_fresh_p.w[0], my_fresh_p.w[1:]
    #print(my_fresh_p.w)
    slope = - my_fresh_p.w[0] / my_fresh_p.w[1]
    intercept = my_fresh_p.w[0] / my_fresh_p.w[1]
    intercept = bias / weight[1]

    #ax.plot(boundary_x := np.linspace(0, 2, 2), slope * boundary_x + intercept)
    #ax.arrow(bias/weight[0], 0, weight[0], weight[1], width=.03, length_includes_head=True)

    plt.show()


    error = 0;
    number_of_predictions = y_test.shape[0]
    for d in range(number_of_predictions):
        if y_predictions[d] != y_test[d]:
            error = error + 1
    print('---Finished evaluation---')
    accuracy = (1 - error/number_of_predictions) * 100
    print('Accuracy: ' + str(accuracy) + '%')


#aufgabe_01()
aufgabe_02_03()
aufgabe_04()
#visualisierung_mit_bias()