from random import randint

import numpy as np
from setuptools._vendor.more_itertools import seekable
from setuptools._vendor.more_itertools.more import _sample_weighted


class Perzeptron:
    global w    #weight
    global r    #lernrate
    global b    #liasterm
    def __init__(self, w_init, learn_rate):
        self.w = w_init
        self.r = learn_rate
        self.b = 0;
        print('Perzeptron has been initialised with initial weight of ' + str(self.w)
              + ' and initial learn rate ' + str(self.r) + '...')

    def infer(self, x):
        """
        INPUT
        x -> Eingabe Vektor
        RETURN
        y -> Lables in das Perzeptron die Eingabevektoren laut seinen internen Gewichten stecken würde
        """
        y = np.zeros(x.shape[1])
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
        for k in range(x.shape[1]):#Schnappt sich Eizelne Zeilen aus der Matrix x. Die einzelnen Zeilen haben n Spalten/Features
            self.weight_new(x[:, k], y[k])#TODO: So richtig? Müssen hier Zeilen oder doch Spalten von X verwendet werden?
            print('New w is now:' + str(self.w))

    @staticmethod
    def generate(number_of_features, dimension):
        """Soll synthetische Daten generieren!"""
        """Matrix mit n Feature vielen Zeilen + 1 Zeile für Lables, Dimension d/Anzahl der Spalten Heißt Anzahl der Eingabevektoren"""
        #w_init = np.array([randint(-100, 100), randint(-100, 100)])
        #Winit
        w_init = np.zeros(number_of_features)
        matrix = np.zeros((number_of_features + 1, dimension))# + 1 für lables
        for n in range(number_of_features):
            w_init[n] = np.array([randint(-100, 100)])

        p = Perzeptron(w_init, 0.001)

        for d in range(dimension):
            for n in range(number_of_features + 1):
                if n == number_of_features:
                    matrix[n, d] = p.activating_function(matrix[:number_of_features, d])
                else:
                    matrix[n, d] = randint(-100, 100)
        return matrix
