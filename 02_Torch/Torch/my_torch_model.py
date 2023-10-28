import torch


class Perzeptron(torch.nn.Module):
    global weight
    def fit(self, xs, ys, epochs, with_bias, mute):
        self.mute = mute
        self.with_bias = with_bias
        if with_bias:
            xs = self.to_affine(xs)

        self.perceptron_fit_simd_minibatch(xs, ys, batchsize=epochs)

    def perceptron_fit_simd_minibatch(self, xs, ys, learn_rate=0.01, batchsize=1):
        self.weight = torch.ones_like(xs[0]) #torch.float64??
        accumulate_weight = self.weight
        for batch in range(len(xs)//batchsize):
            xs_batch = xs[batch*batchsize:(batch+1)*batchsize]
            ys_batch = ys[batch*batchsize:(batch+1)*batchsize]
            predictions = self.perceptron_predict_simd2(accumulate_weight, xs_batch)
            y_pred_vector = ys_batch - predictions
            aenderungs_vektoren = torch.ones_like(xs_batch)
            for row in range(xs_batch.shape[0]):
                aenderungs_vektoren[row, :] = y_pred_vector[row] * xs_batch[row, :]
            summe_aenderungen_in_x = torch.sum(aenderungs_vektoren, axis=0)
            self.weight += summe_aenderungen_in_x * learn_rate
            #self.show_information("Trained a batch!")

    def to_affine(self, x):
        #import numpy as np
        #print("This Matrix: ")
        #print(np.append(np.ones_like(x[:,0]).reshape(-1,1), x.numpy(), axis=1))
        #print("Should look like: ")
        affine_tensor = torch.cat([torch.ones_like(x[:,0]).reshape(-1,1), x], 1)
        #print(affine_tensor)
        return affine_tensor

    def perceptron_predict_simd2(self, weight_vector, xs):
        return torch.where(torch.tensordot(xs, weight_vector, 1) > 0, 1, 0)#Removed ", axes=1" when converting to torch model

    def get_decision_boundary(self):
        if(self.with_bias):
            bias, w = - self.weight[0], self.weight[1:]
            slope = - w[0] / w[1]
            intercept = bias / w[1]
        else:
            slope = - self.weight[0] / self.weight[1]
            intercept = 0
        return slope, intercept