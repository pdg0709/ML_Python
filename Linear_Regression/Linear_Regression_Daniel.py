from Boston_Data import model_data as lrdata
import numpy as np

class LinearRegression:

    def __init__(self, learn_rate=0.2, iter_times=200000, error=1e-9):
        self.learn_rate = learn_rate
        self.iter_times = iter_times
        self.error = error

    def Trans(self, xdata):
        one1 = np.ones(len(xdata))
        xta = np.append(xdata, one1.reshape(-1, 1), axis=1)
        return xta

    def Gradent(self, xdata, ydata):
        xdata = self.Trans(xdata)
        self.weights = np.zeros((len(xdata[0]), 1))
        