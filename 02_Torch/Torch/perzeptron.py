from copyreg import add_extension
from random import randint

import numpy as np

class Perzeptron:
    global weight    #weight
    global mute
    global with_bias
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
        self.weight = np.ones_like(xs[0], dtype="float64")
        accumulate_weight = self.weight
        self.show_information("Starting minibatch training...")
        for batch in range(len(xs)//batchsize):
            xs_batch = xs[batch*batchsize:(batch+1)*batchsize]
            ys_batch = ys[batch*batchsize:(batch+1)*batchsize]
            #print("Batches: \n" + str(xs_batch) + str(ys_batch))
            #print("Predictions: \n" + str(self.perceptron_predict_simd2(accumulate_weight, xs_batch)))
            predictions = self.perceptron_predict_simd2(accumulate_weight, xs_batch)
            #print("Y-Y_pred Vector: " + str(ys_batch - predictions))
            y_pred_vector = ys_batch - predictions
            aenderungs_vektoren = np.ones_like(xs_batch)
            for row in range(xs_batch.shape[0]):
                #print("Berechne Änderungsvektor: \n")
                #print(str(y_pred_vector[row]) + "* " + str(xs_batch[row, :]))
                aenderungs_vektoren[row, :] = y_pred_vector[row] * xs_batch[row, :]
            #print("Aenderungs vektoren: " + str(aenderungs_vektoren))
            summe_aenderungen_in_x = np.sum(aenderungs_vektoren, axis=0)
            #print("Summer aller änderungen: " + str(summe_aenderungen_in_x))
            self.weight += summe_aenderungen_in_x * learn_rate
            self.show_information("Trained a batch!")

            """accumulate_weight += learn_rate * np.sum( (ys_batch \
                                                   - self.perceptron_predict_simd2(accumulate_weight, xs_batch))
                                                  * xs_batch,
                                                  axis=0)"""


    def perceptron_predict_simd2(self, weight_vector, xs):
        return np.where(np.tensordot(xs, weight_vector, axes=1) > 0, 1, 0)

    def get_decision_boundary(self):
        if(self.with_bias):
            bias, w = - self.weight[0], self.weight[1:]
            slope = - w[0] / w[1]
            intercept = bias / w[1]
        else:
            slope = - self.weight[0] / self.weight[1]
            intercept = 0
        return slope, intercept

    def fit(self, xs, ys, epochs, with_bias, mute):
        self.mute = mute
        self.with_bias = with_bias
        if with_bias:
            xs = self.to_affine(xs)

        if epochs == 1:
            self.train(xs, ys)
        else:
            self.perceptron_fit_simd_minibatch(xs, ys, batchsize=epochs)