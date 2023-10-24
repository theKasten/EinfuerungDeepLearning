import numpy as np
from setuptools._vendor.more_itertools.more import _sample_weighted


class Perzeptron:
    global w
    global r
    def __init__(self, w_init, learn_rate):
        self.w = w_init
        self.r = learn_rate
        print('Perzeptron has been initialised with initial weight of ' + str(self.w)
              + ' and initial learn rate ' + str(self.r) + '...')

    def infer(self, x):
        """
        INPUT
        x -> Eingabe Vektor
        RETURN
        Soll Klasse ausgeben in das Perzeptron den Eingabevektor laut seinen internen Gewichten stecken würde
        """
        return self.activating_function(x)

    def activating_function(self, x):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        w -> Gewicht
        :return:
        Vorhersage der Binären klassifikation von x mit gewicht w
        (Also gehört x Klasse 0 oder 1 an)
        """
        if np.dot(self.w, x) > 0:
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
        #x = np.array(x_in)
        #print("Shape x:" + str(x.shape))
        for k in range(x.shape[1]):
            self.weight_new(x[k, :], y[k])#TODO: So richtig? Müssen hier Reihen oder doch Spalten von X verwendet werden?
            print('New w is now:' + str(self.w))

