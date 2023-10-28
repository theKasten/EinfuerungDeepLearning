from random import randint

import numpy as np

class Perzeptron:
    global weight    #weight
    global mute
    #global r    #lernrate

    @staticmethod
    def to_affine(x):
        return np.append(np.ones_like(x[:,0]).reshape(-1,1), x, axis=1)
    def __init__(self):
        self.mute = False
        self.weight = None
        #self.r = None
        #self.w = np.ones(1, dtype="float64")
        #print('Perzeptron has been initialised with initial weight of ' + str(self.w)
        #      + ' and initial learn rate ' + str(self.r) + '...')

    def infer(self, x):
        """
        INPUT
        x → Eingabe Vektor
        RETURN
        y → Lables in das Perzeptron die Eingabevektoren laut seinen internen Gewichten stecken würde
        """
        y = np.zeros(x.shape[1], dtype=int)
        for k in range(x.shape[1]):
            y[k] = self.activating_function(x[:, k])

        return y

    def activating_function(self, x):
        """
        INPUT:
        x → Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        w → Gewich
        :return:
        Vorhersage der Binären klassifikation von x mit gewicht w
        (Also gehört x Klasse 0 oder 1 an)
        """
        #print("w, x: " + str(self.weight) + str(x))
        if np.dot(self.weight, x) > 0:
            #print("Element in klass 1")
            return 1
        else:
            #print("Element in klass 0")
            return 0
        # Besser: #return np.where(np.tensordot(x, self.w, axes=1) > 0, 1, 0)

    def weight_new(self, x, y, learn_rate):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        y -> Lable
        w(z)? -> Gewicht??
        r -> lernrate
        :return:
        """
        self.show_information("Calc new weight based on: ")
        self.weight = self.weight/np.linalg.norm(self.weight)
        self.show_information("Weight now normed: ")
        self.weight += learn_rate * (y - self.activating_function(x)) * x#Hier += vergessen!!
        self.show_information("weight update done: ")

    def train(self, xs, ys, learn_rate = 0.01):
        #print("Shape x:" + str(x.shape))
        self.weight = np.ones_like(xs[0], dtype="float64")
        for x, y in zip(xs, ys):
            #self.show_information()
            #print("Now Calc new weight")
            self.weight_new(x, y, learn_rate)
            #print("Done with new weight")
            #self.show_information()
            #self.w = self.w/np.linalg.norm(self.w)#Normieren damit gewicht nicht zu klein wird

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

        p = Perzeptron()

        for d in range(dimension):
            for n in range(number_of_features + 1):
                if n == number_of_features:
                    matrix[n, d] = p.activating_function(matrix[:number_of_features, d])
                else:
                    matrix[n, d] = randint(-100, 100)
        return matrix

    def show_information(self, information):
        if not self.mute:
            print(information)
            print('Perzeptron has weight of ' + str(self.weight)
                  + ' and learn rate ' + str(None) + '...')
        #do nothing

    """def perceptron_fit_simd_minibatch(self, xs, ys, learn_rate=0.01, batchsize=1):
        weight_vector=np.ones_like(xs[0], dtype='float64')
        for batch in range(len(xs)//batchsize):
            xs_batch = xs[batch*batchsize:(batch+1)*batchsize]
            ys_batch = ys[batch*batchsize:(batch+1)*batchsize]
            self.weight += learn_rate * np.sum((ys_batch \
                                                - self.perceptron_predict_simd2(xs_batch))
                                               * xs_batch,
                                               axis=0)"""

    def perceptron_fit_simd_minibatch(self, xs, ys, learn_rate=0.01, batchsize=1):
        weight_vector = np.ones_like(xs[0], dtype="float64")
        for batch in range(len(xs)//batchsize):
            xs_batch = xs[batch*batchsize:(batch+1)*batchsize]
            ys_batch = ys[batch*batchsize:(batch+1)*batchsize]
            weight_vector += learn_rate * np.sum( (ys_batch \
                                                   - self.perceptron_predict_simd2(weight_vector, xs_batch))
                                                  * xs_batch,
                                                  axis=0)
        self.weight = weight_vector

    def perceptron_predict_simd2(self, weight_vector, xs):
        return np.where(np.tensordot(xs, weight_vector, axes=1) > 0, 1, 0)