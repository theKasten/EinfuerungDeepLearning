import numpy as np
class Perzeptron:
    def f(self):
        print('Do nothing')
    def activating_function(self, x, w):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        w -> Gewicht
        :return:
        Vorhersage der Binären klassifikation von x mit gewicht w
        (Also gehört x Klasse 0 oder 1 an)
        """
        if np.dot(w, x) > 0:
            #print("Element in klass 1")
            return 1
        else:
            #print("Element in klass 0")
            return 0

    def weight_new(self, x, y, w, r):
        """
        INPUT:
        x -> Eingabe Vektor (einzelner Zeile mit Anzahl der Features(Spalten) vielen ausprägungen)
        y -> Lable
        w(z)? -> Gewicht??
        r -> lernrate
        :return:
        """
        #return w + np.dot(r * (y - self.activating_function(w, x)) , x)#TODO: So richtig??
        return w + r * (y - self.activating_function(w, x)) * x#TODO: So richtig??
