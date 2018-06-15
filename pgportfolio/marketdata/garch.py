from __future__ import absolute_import, print_function, division
import numpy as np
import copy
from pgportfolio.tools.configprocess import parse_time, check_input_same
from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.tools.configprocess import load_config
from functools import partial
import math
import cvxopt
from functools import partial
import scipy
from scipy import stats

import pandas as pd

import matplotlib.pyplot as plt




class Garch:
    def __init__(self, config):
        self.__counter = 0
        self.__start = parse_time(config["input"]["start_date"])
        self.__end = parse_time(config["input"]["end_date"])
        self.__number_of_coins = config["input"]["coin_number"]
        self.__batch_size = config["training"]["batch_size"]
        self.__window_size = config["input"]["window_size"]+1
        span = self.__end - self.__start
        self.__end = self.__end - config["input"]["test_portion"] * span
        config2 = copy.deepcopy(config)
        config2["input"]["global_period"] = 300
        self._matrix2 = DataMatrices.create_from_config(config2)
        self.__weightgarch = pd.DataFrame(index=range(0, self.__batch_size),
                                              columns=range(0, self.__number_of_coins))
        self.__weightgarch = self.__weightgarch.fillna(1.0 / self.__number_of_coins)
        training_set = self._matrix2.get_training_set()
        set = training_set["X"]

        #good times sets for 3, 26: 1 and 2 are not functioning
        set5 = set[-5000:, 0, :, 0]
        set3 = set[-3000:, 0, :, 0]
        set2 = set[-2000:, 0, :, 0]
        #self.__abcdefg = set[:, 0]
        self.__lastvalue1 = set5[0, 0]
        self.__lastvalue2 = set5[0, 1]
        self.__lastvalue3 = set2[0, 2]
        self.__lastvalue4 = set5[0, 3]
        self.__lastvalue5 = set3[0, 4]
        self.__lastvalue6 = set3[0, 5]
        self.__lastvalue7 = set3[0, 6]
        self.__lastvalue8 = set5[0, 7]
        self.__lastvalue9 = set5[0, 8]
        self.__lastvalue10 = set5[0, 9]
        self.__lastvalue11 = set3[0, 10]
        #good times set5, bad times set3
        #logreturns1 = np.log(set3[1:, 0] / set3[:-1, 0])
        logreturns1 = np.log(set5[1:, 0] / set5[:-1, 0])
        self.__lastsigma1 = np.sqrt(np.mean(logreturns1 ** 2))
        self.__lastlogreturn1 = logreturns1[-1]
        #good times set5, bad times set3
        #logreturns2 = np.log(set3[1:, 1] / set3[:-1, 1])
        logreturns2 = np.log(set5[1:, 1] / set5[:-1, 1])
        self.__lastsigma2 = np.sqrt(np.mean(logreturns2 ** 2))
        self.__lastlogreturn2 = logreturns2[-1]
        logreturns3 = np.log(set2[1:, 2] / set2[:-1, 2])
        self.__lastsigma3 = np.sqrt(np.mean(logreturns3 ** 2))
        self.__lastlogreturn3 = logreturns3[-1]
        logreturns4 = np.log(set5[1:, 3] / set5[:-1, 3])
        self.__lastsigma4 = np.sqrt(np.mean(logreturns4 ** 2))
        self.__lastlogreturn4 = logreturns4[-1]
        logreturns5 = np.log(set3[1:, 4] / set3[:-1, 4])
        self.__lastsigma5 = np.sqrt(np.mean(logreturns5 ** 2))
        self.__lastlogreturn5 = logreturns5[-1]
        logreturns6 = np.log(set3[1:, 5] / set3[:-1, 5])
        self.__lastsigma6 = np.sqrt(np.mean(logreturns6 ** 2))
        self.__lastlogreturn6 = logreturns6[-1]
        logreturns7 = np.log(set3[1:, 6] / set3[:-1, 6])
        self.__lastsigma7 = np.sqrt(np.mean(logreturns7 ** 2))
        self.__lastlogreturn7 = logreturns7[-1]
        logreturns8 = np.log(set5[1:, 7] / set5[:-1, 7])
        self.__lastsigma8 = np.sqrt(np.mean(logreturns8 ** 2))
        self.__lastlogreturn8 = logreturns8[-1]
        logreturns9 = np.log(set5[1:, 8] / set5[:-1, 8])
        self.__lastsigma9 = np.sqrt(np.mean(logreturns9 ** 2))
        self.__lastlogreturn9 = logreturns9[-1]
        logreturns10 = np.log(set5[1:, 9] / set5[:-1, 9])
        self.__lastsigma10 = np.sqrt(np.mean(logreturns10 ** 2))
        self.__lastlogreturn10 = logreturns10[-1]
        logreturns11 = np.log(set3[1:, 10] / set3[:-1, 10])
        self.__lastsigma11 = np.sqrt(np.mean(logreturns11 ** 2))
        self.__lastlogreturn11 = logreturns11[-1]

        self.__firstsigma1 = self.__lastsigma1
        self.__firstsigma2 = self.__lastsigma2
        self.__firstsigma3 = self.__lastsigma3
        self.__firstsigma4 = self.__lastsigma4
        self.__firstsigma5 = self.__lastsigma5
        self.__firstsigma6 = self.__lastsigma6
        self.__firstsigma7 = self.__lastsigma7
        self.__firstsigma8 = self.__lastsigma8
        self.__firstsigma9 = self.__lastsigma9
        self.__firstsigma10 = self.__lastsigma10
        self.__firstsigma11 = self.__lastsigma11

        self.__firstlogreturn1 = self.__lastlogreturn1
        self.__firstlogreturn2 = self.__lastlogreturn2
        self.__firstlogreturn3 = self.__lastlogreturn3
        self.__firstlogreturn4 = self.__lastlogreturn4
        self.__firstlogreturn5 = self.__lastlogreturn5
        self.__firstlogreturn6 = self.__lastlogreturn6
        self.__firstlogreturn7 = self.__lastlogreturn7
        self.__firstlogreturn8 = self.__lastlogreturn8
        self.__firstlogreturn9 = self.__lastlogreturn9
        self.__firstlogreturn10 = self.__lastlogreturn10
        self.__firstlogreturn11 = self.__lastlogreturn11

        self.__firstlastvalue1 = self.__lastvalue1
        self.__firstlastvalue2 = self.__lastvalue2
        self.__firstlastvalue3 = self.__lastvalue3
        self.__firstlastvalue4 = self.__lastvalue4
        self.__firstlastvalue5 = self.__lastvalue5
        self.__firstlastvalue6 = self.__lastvalue6
        self.__firstlastvalue7 = self.__lastvalue7
        self.__firstlastvalue8 = self.__lastvalue8
        self.__firstlastvalue9 = self.__lastvalue9
        self.__firstlastvalue10 = self.__lastvalue10
        self.__firstlastvalue11 = self.__lastvalue11

        self.__theta1 = self.negative_log_likelihood(logreturns1, (1, 0.5, 0.5))
        self.__theta1 = self.fitting(logreturns1)
        print(self.__theta1)
        self.__theta2 = self.fitting(logreturns2)
        print(self.__theta2)
        self.__theta3 = self.fitting(logreturns3)
        print(self.__theta3)
        self.__theta4 = self.fitting(logreturns4)
        print(self.__theta4)
        self.__theta5 = self.fitting(logreturns5)
        print(self.__theta5)
        self.__theta6 = self.fitting(logreturns6)
        print(self.__theta6)
        self.__theta7 = self.fitting(logreturns7)
        print(self.__theta7)
        self.__theta8 = self.fitting(logreturns8)
        print(self.__theta8)
        self.__theta9 = self.fitting(logreturns9)
        print(self.__theta9)
        self.__theta10 = self.fitting(logreturns10)
        print(self.__theta10)
        self.__theta11 = self.fitting(logreturns11)
        print(self.__theta11)

    def simulate(self):
        if self.__counter % 500 == 0:
            self.__weightgarch = pd.DataFrame(index=range(0, self.__batch_size),
                                              columns=range(0, self.__number_of_coins))
            self.__weightgarch = self.__weightgarch.fillna(1.0 / self.__number_of_coins)
            #print(self.__counter)
            self.__lastsigma1 = self.__firstsigma1
            self.__lastsigma2 = self.__firstsigma2
            self.__lastsigma3 = self.__firstsigma3
            self.__lastsigma4 = self.__firstsigma4
            self.__lastsigma5 = self.__firstsigma5
            self.__lastsigma6 = self.__firstsigma6
            self.__lastsigma7 = self.__firstsigma7
            self.__lastsigma8 = self.__firstsigma8
            self.__lastsigma9 = self.__firstsigma9
            self.__lastsigma10 = self.__firstsigma10
            self.__lastsigma11 = self.__firstsigma11

            self.__lastlogreturn1 = self.__firstlogreturn1
            self.__lastlogreturn2 = self.__firstlogreturn2
            self.__lastlogreturn3 = self.__firstlogreturn3
            self.__lastlogreturn4 = self.__firstlogreturn4
            self.__lastlogreturn5 = self.__firstlogreturn5
            self.__lastlogreturn6 = self.__firstlogreturn6
            self.__lastlogreturn7 = self.__firstlogreturn7
            self.__lastlogreturn8 = self.__firstlogreturn8
            self.__lastlogreturn9 = self.__firstlogreturn9
            self.__lastlogreturn10 = self.__firstlogreturn10
            self.__lastlogreturn11 = self.__firstlogreturn11

            self.__lastvalue1 = self.__firstlastvalue1
            self.__lastvalue2 = self.__firstlastvalue2
            self.__lastvalue3 = self.__firstlastvalue3
            self.__lastvalue4 = self.__firstlastvalue4
            self.__lastvalue5 = self.__firstlastvalue5
            self.__lastvalue6 = self.__firstlastvalue6
            self.__lastvalue7 = self.__firstlastvalue7
            self.__lastvalue8 = self.__firstlastvalue8
            self.__lastvalue9 = self.__firstlastvalue9
            self.__lastvalue10 = self.__firstlastvalue10
            self.__lastvalue11 = self.__firstlastvalue11

        self.__counter = self.__counter + 1

        X1, self.__lastsigma1, self.__lastlogreturn1 = \
            self.simulate_GARCH(6*(self.__batch_size+self.__window_size), self.__theta1[0],
            self.__theta1[1], self.__theta1[2], self.__lastsigma1, self.__lastlogreturn1, self.__lastvalue1)
        X1max = np.zeros(self.__window_size)
        X1min = np.zeros(self.__window_size)
        X1closing = np.zeros(self.__window_size)
        X1max[0] = np.amax(np.append(X1[range(0, 5)], self.__lastvalue1))
        X1min[0] = np.amin(np.append(X1[range(0, 5)], self.__lastvalue1))
        X1closing[0] = X1[5]
        self.__lastvalue1 = X1[-1]
        for i in range(1, self.__window_size):
            X1max[i] = np.amax(X1[range(i*6-1, (i+1)*6)])
            X1min[i] = np.amin(X1[range(i*6-1, (i+1)*6)])
            X1closing[i] = X1[(i+1)*6-1]
        #print(np.shape(X1closing))
        X1batch = np.stack((X1closing, X1max, X1min), axis=0)
        #print(np.shape(X1batch))
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X1max[i] = np.amax(X1[range(6*j+i * 6 - 1, 6*j+(i + 1) * 6)])
                X1min[i] = np.amin(X1[range(6*j+i * 6 - 1, 6*j+(i + 1) * 6)])
                X1closing[i] = X1[6*j+i * 6 - 1]
            X1batchnew = np.stack((X1closing, X1max, X1min), axis=0)
            if j == 1:
                X1batch = np.stack((X1batch, X1batchnew), axis=0)
            else:
                X1batch = np.append(X1batch, [X1batchnew], axis=0)


        X2, self.__lastsigma2, self.__lastlogreturn2 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta2[0],
                                self.__theta2[1], self.__theta2[2], self.__lastsigma2, self.__lastlogreturn2,
                                self.__lastvalue2)
        X2max = np.zeros(self.__window_size)
        X2min = np.zeros(self.__window_size)
        X2closing = np.zeros(self.__window_size)
        X2max[0] = np.amax(np.append(X2[range(0, 5)], self.__lastvalue2))
        X2min[0] = np.amin(np.append(X2[range(0, 5)], self.__lastvalue2))
        X2closing[0] = X2[5]
        self.__lastvalue2 = X2[-1]
        for i in range(1, self.__window_size):
            X2max[i] = np.amax(X2[range(i * 6 - 1, (i + 1) * 6)])
            X2min[i] = np.amin(X2[range(i * 6 - 1, (i + 1) * 6)])
            X2closing[i] = X2[(i+1) * 6 - 1]
        X2batch = np.stack((X2closing, X2max, X2min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X2max[i] = np.amax(X2[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X2min[i] = np.amin(X2[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X2closing[i] = X2[6 * j + i * 6 - 1]
            X2batchnew = np.stack((X2closing, X2max, X2min), axis=0)
            if j == 1:
                X2batch = np.stack((X2batch, X2batchnew), axis=0)
            else:
                X2batch = np.append(X2batch, [X2batchnew], axis=0)



        X3, self.__lastsigma3, self.__lastlogreturn3 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta3[0],
                                self.__theta3[1], self.__theta3[2], self.__lastsigma3, self.__lastlogreturn3,
                                self.__lastvalue3)
        X3max = np.zeros(self.__window_size)
        X3min = np.zeros(self.__window_size)
        X3closing = np.zeros(self.__window_size)
        X3max[0] = np.amax(np.append(X3[range(0, 5)], self.__lastvalue3))
        X3min[0] = np.amin(np.append(X3[range(0, 5)], self.__lastvalue3))
        X3closing[0] = X3[5]
        self.__lastvalue3 = X3[-1]
        for i in range(1, self.__window_size):
            X3max[i] = np.amax(X3[range(i * 6 - 1, (i + 1) * 6)])
            X3min[i] = np.amin(X3[range(i * 6 - 1, (i + 1) * 6)])
            X3closing[i] = X3[(i+1) * 6 - 1]
        X3batch = np.stack((X3closing, X3max, X3min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X3max[i] = np.amax(X3[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X3min[i] = np.amin(X3[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X3closing[i] = X3[6 * j + i * 6 - 1]
            X3batchnew = np.stack((X3closing, X3max, X3min), axis=0)
            if j == 1:
                X3batch = np.stack((X3batch, X3batchnew), axis=0)
            else:
                X3batch = np.append(X3batch, [X3batchnew], axis=0)





        X4, self.__lastsigma4, self.__lastlogreturn4 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta4[0],
                                self.__theta4[1], self.__theta4[2], self.__lastsigma4, self.__lastlogreturn4,
                                self.__lastvalue4)
        X4max = np.zeros(self.__window_size)
        X4min = np.zeros(self.__window_size)
        X4closing = np.zeros(self.__window_size)
        X4max[0] = np.amax(np.append(X4[range(0, 5)], self.__lastvalue4))
        X4min[0] = np.amin(np.append(X4[range(0, 5)], self.__lastvalue4))
        X4closing[0] = X4[5]
        self.__lastvalue4 = X4[-1]
        for i in range(1, self.__window_size):
            X4max[i] = np.amax(X4[range(i * 6 - 1, (i + 1) * 6)])
            X4min[i] = np.amin(X4[range(i * 6 - 1, (i + 1) * 6)])
            X4closing[i] = X4[(i+1) * 6 - 1]
        X4batch = np.stack((X4closing, X4max, X4min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X4max[i] = np.amax(X4[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X4min[i] = np.amin(X4[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X4closing[i] = X4[6 * j + i * 6 - 1]
            X4batchnew = np.stack((X4closing, X4max, X4min), axis=0)
            if j == 1:
                X4batch = np.stack((X4batch, X4batchnew), axis=0)
            else:
                X4batch = np.append(X4batch, [X4batchnew], axis=0)




        X5, self.__lastsigma5, self.__lastlogreturn5 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta5[0],
                                self.__theta5[1], self.__theta5[2], self.__lastsigma5, self.__lastlogreturn5,
                                self.__lastvalue5)
        X5max = np.zeros(self.__window_size)
        X5min = np.zeros(self.__window_size)
        X5closing = np.zeros(self.__window_size)
        X5max[0] = np.amax(np.append(X5[range(0, 5)], self.__lastvalue5))
        X5min[0] = np.amin(np.append(X5[range(0, 5)], self.__lastvalue5))
        X5closing[0] = X5[5]
        self.__lastvalue5 = X1[-1]
        for i in range(1, self.__window_size):
            X5max[i] = np.amax(X5[range(i * 6 - 1, (i + 1) * 6)])
            X5min[i] = np.amin(X5[range(i * 6 - 1, (i + 1) * 6)])
            X5closing[i] = X5[(i+1) * 6 - 1]
        X5batch = np.stack((X5closing, X5max, X5min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X5max[i] = np.amax(X5[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X5min[i] = np.amin(X5[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X5closing[i] = X5[6 * j + i * 6 - 1]
            X5batchnew = np.stack((X5closing, X5max, X5min), axis=0)
            if j == 1:
                X5batch = np.stack((X5batch, X5batchnew), axis=0)
            else:
                X5batch = np.append(X5batch, [X5batchnew], axis=0)






        X6, self.__lastsigma6, self.__lastlogreturn6 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta6[0],
                                self.__theta6[1], self.__theta6[2], self.__lastsigma6, self.__lastlogreturn6,
                                self.__lastvalue6)
        X6max = np.zeros(self.__window_size)
        X6min = np.zeros(self.__window_size)
        X6closing = np.zeros(self.__window_size)
        X6max[0] = np.amax(np.append(X6[range(0, 5)], self.__lastvalue6))
        X6min[0] = np.amin(np.append(X6[range(0, 5)], self.__lastvalue6))
        X6closing[0] = X6[5]
        self.__lastvalue6 = X6[-1]
        for i in range(1, self.__window_size):
            X6max[i] = np.amax(X6[range(i * 6 - 1, (i + 1) * 6)])
            X6min[i] = np.amin(X6[range(i * 6 - 1, (i + 1) * 6)])
            X6closing[i] = X6[(i+1) * 6 - 1]
        X6batch = np.stack((X6closing, X6max, X6min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X6max[i] = np.amax(X6[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X6min[i] = np.amin(X6[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X6closing[i] = X6[6 * j + i * 6 - 1]
            X6batchnew = np.stack((X6closing, X6max, X6min), axis=0)
            if j == 1:
                X6batch = np.stack((X6batch, X6batchnew), axis=0)
            else:
                X6batch = np.append(X6batch, [X6batchnew], axis=0)

        X7, self.__lastsigma7, self.__lastlogreturn7 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta7[0],
                                self.__theta7[1], self.__theta7[2], self.__lastsigma7, self.__lastlogreturn7,
                                self.__lastvalue7)
        X7max = np.zeros(self.__window_size)
        X7min = np.zeros(self.__window_size)
        X7closing = np.zeros(self.__window_size)
        X7max[0] = np.amax(np.append(X7[range(0, 5)], self.__lastvalue7))
        X7min[0] = np.amin(np.append(X7[range(0, 5)], self.__lastvalue7))
        X7closing[0] = X7[5]
        self.__lastvalue7 = X7[-1]
        for i in range(1, self.__window_size):
            X7max[i] = np.amax(X7[range(i * 6 - 1, (i + 1) * 6)])
            X7min[i] = np.amin(X7[range(i * 6 - 1, (i + 1) * 6)])
            X7closing[i] = X7[(i+1) * 6 - 1]
        X7batch = np.stack((X7closing, X7max, X7min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X7max[i] = np.amax(X7[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X7min[i] = np.amin(X7[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X7closing[i] = X7[6 * j + i * 6 - 1]
            X7batchnew = np.stack((X7closing, X7max, X7min), axis=0)
            if j == 1:
                X7batch = np.stack((X7batch, X7batchnew), axis=0)
            else:
                X7batch = np.append(X7batch, [X7batchnew], axis=0)



        X8, self.__lastsigma8, self.__lastlogreturn8 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta8[0],
                                self.__theta8[1], self.__theta8[2], self.__lastsigma8, self.__lastlogreturn8,
                                self.__lastvalue8)
        X8max = np.zeros(self.__window_size)
        X8min = np.zeros(self.__window_size)
        X8closing = np.zeros(self.__window_size)
        X8max[0] = np.amax(np.append(X8[range(0, 5)], self.__lastvalue8))
        X8min[0] = np.amin(np.append(X8[range(0, 5)], self.__lastvalue8))
        X8closing[0] = X8[5]
        self.__lastvalue8 = X8[-1]
        for i in range(1, self.__window_size):
            X8max[i] = np.amax(X8[range(i * 6 - 1, (i + 1) * 6)])
            X8min[i] = np.amin(X8[range(i * 6 - 1, (i + 1) * 6)])
            X8closing[i] = X8[(i+1) * 6 - 1]
        X8batch = np.stack((X8closing, X8max, X8min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X8max[i] = np.amax(X8[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X8min[i] = np.amin(X8[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X8closing[i] = X8[6 * j + i * 6 - 1]
            X8batchnew = np.stack((X8closing, X8max, X8min), axis=0)
            if j == 1:
                X8batch = np.stack((X8batch, X8batchnew), axis=0)
            else:
                X8batch = np.append(X8batch, [X8batchnew], axis=0)




        X9, self.__lastsigma9, self.__lastlogreturn9 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta9[0],
                                self.__theta9[1], self.__theta9[2], self.__lastsigma9, self.__lastlogreturn9,
                                self.__lastvalue9)
        X9max = np.zeros(self.__window_size)
        X9min = np.zeros(self.__window_size)
        X9closing = np.zeros(self.__window_size)
        X9max[0] = np.amax(np.append(X9[range(0, 5)], self.__lastvalue9))
        X9min[0] = np.amin(np.append(X9[range(0, 5)], self.__lastvalue9))
        X9closing[0] = X9[5]
        self.__lastvalue9 = X9[-1]
        for i in range(1, self.__window_size):
            X9max[i] = np.amax(X9[range(i * 6 - 1, (i + 1) * 6)])
            X9min[i] = np.amin(X9[range(i * 6 - 1, (i + 1) * 6)])
            X9closing[i] = X9[(i+1) * 6 - 1]
        X9batch = np.stack((X9closing, X9max, X9min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X9max[i] = np.amax(X9[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X9min[i] = np.amin(X9[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X9closing[i] = X9[6 * j + i * 6 - 1]
            X9batchnew = np.stack((X9closing, X9max, X9min), axis=0)
            if j == 1:
                X9batch = np.stack((X9batch, X9batchnew), axis=0)
            else:
                X9batch = np.append(X9batch, [X9batchnew], axis=0)

        X10, self.__lastsigma10, self.__lastlogreturn10 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta10[0],
                                self.__theta10[1], self.__theta10[2], self.__lastsigma10, self.__lastlogreturn10,
                                self.__lastvalue10)
        X10max = np.zeros(self.__window_size)
        X10min = np.zeros(self.__window_size)
        X10closing = np.zeros(self.__window_size)
        X10max[0] = np.amax(np.append(X10[range(0, 5)], self.__lastvalue10))
        X10min[0] = np.amin(np.append(X10[range(0, 5)], self.__lastvalue10))
        X10closing[0] = X10[5]
        self.__lastvalue10 = X10[-1]
        for i in range(1, self.__window_size):
            X10max[i] = np.amax(X10[range(i * 6 - 1, (i + 1) * 6)])
            X10min[i] = np.amin(X10[range(i * 6 - 1, (i + 1) * 6)])
            X10closing[i] = X10[(i+1) * 6 - 1]
        X10batch = np.stack((X10closing, X10max, X10min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X10max[i] = np.amax(X10[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X10min[i] = np.amin(X10[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X10closing[i] = X10[6 * j + i * 6 - 1]
            X10batchnew = np.stack((X10closing, X10max, X10min), axis=0)
            if j == 1:
                X10batch = np.stack((X10batch, X10batchnew), axis=0)
            else:
                X10batch = np.append(X10batch, [X10batchnew], axis=0)




        X11, self.__lastsigma11, self.__lastlogreturn11 = \
            self.simulate_GARCH(6 * (self.__batch_size + self.__window_size), self.__theta11[0],
                                self.__theta11[1], self.__theta11[2], self.__lastsigma11, self.__lastlogreturn11,
                                self.__lastvalue11)
        X11max = np.zeros(self.__window_size)
        X11min = np.zeros(self.__window_size)
        X11closing = np.zeros(self.__window_size)
        X11max[0] = np.amax(np.append(X11[range(0, 5)], self.__lastvalue11))
        X11min[0] = np.amin(np.append(X11[range(0, 5)], self.__lastvalue11))
        X11closing[0] = X11[5]
        self.__lastvalue11 = X11[-1]
        for i in range(1, self.__window_size):
            X11max[i] = np.amax(X11[range(i * 6 - 1, (i + 1) * 6)])
            X11min[i] = np.amin(X11[range(i * 6 - 1, (i + 1) * 6)])
            X11closing[i] = X11[(i+1) * 6 - 1]
        X11batch = np.stack((X11closing, X11max, X11min), axis=0)
        for j in range(1, self.__batch_size):
            for i in range(0, self.__window_size):
                X11max[i] = np.amax(X11[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X11min[i] = np.amin(X11[range(6 * j + i * 6 - 1, 6 * j + (i + 1) * 6)])
                X11closing[i] = X11[6 * j + i * 6 - 1]
            X11batchnew = np.stack((X11closing, X11max, X11min), axis=0)
            if j == 1:
                X11batch = np.stack((X11batch, X11batchnew), axis=0)
            else:
                X11batch = np.append(X11batch, [X11batchnew], axis=0)

        Batchnew = np.stack((X1batch,X2batch,X3batch,X4batch,X5batch,X6batch,X7batch,X8batch,X9batch,X10batch,X11batch), axis=2)

        indexs = np.array(range(0, self.__batch_size))
        self.__lastweightgarch = self.__weightgarch.values[indexs, :]
        def setw(w):
            for i in range(0,self.__batch_size):
                self.__weightgarch.iloc[i, :] = w[-1,:]

        Batchnew = np.array(Batchnew, dtype=np.float64)
        X = Batchnew[:, :, :, :-1]
        y = Batchnew[:, :, :, -1] / Batchnew[:, 0, None, :, -2]




        meanofallsigma = np.mean([self.__lastsigma1,self.__lastsigma2, self.__lastsigma3, self.__lastsigma4, self.__lastsigma5, self.__lastsigma6, self.__lastsigma7, self.__lastsigma8, self.__lastsigma9, self.__lastsigma10, self.__lastsigma11])


        if np.isnan(X).any():
            print('Fehler bei X')
        if np.isnan(y).any():
            print('Fehler bei y')
        #print(np.argwhere(np.isnan(X)))
        #print(np.argwhere(np.isnan(y)))
        #print(np.argwhere(np.isnan(self.__lastweightgarch)))
        #return {"X": X, "y": y, "last_w": self.__lastweightgarch, "setw": setw}
        return {"X": X, "y": y, "last_w": self.__lastweightgarch, "setw": setw, "sigma": meanofallsigma}


    def simulate_GARCH(self, T, a0, a1, b1, sigma1, X1, lastvalue):
        # Initialize our values
        X = np.ndarray(T)
        sigma = np.ndarray(T)
        sigma[0] = math.sqrt(a0 + b1*sigma1**2 + a1 * X1 ** 2)

        for t in range(1, T):
            # Draw the next x_t
            X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
            # Draw the next sigma_t
            sigma[t] = math.sqrt(a0 + b1 * sigma[t - 1] ** 2 + a1 * X[t - 1] ** 2)

        X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)
        lastlogreturn = X[-1]
        X = np.exp(X)
        X[0] = X[0]*lastvalue
        for i in range(X.size - 1):
            X[i+1] = X[i+1] * X[i]

        return X, sigma[-1], lastlogreturn
        #return X, sigma1, lastlogreturn

    def compute_squared_sigmas(self, X, initial_sigma, theta):

        a0 = theta[0]
        a1 = theta[1]
        b1 = theta[2]

        T = len(X)
        sigma2 = np.ndarray(T)

        sigma2[0] = initial_sigma ** 2

        for t in range(1, T):
            # Here's where we apply the equation
            sigma2[t] = a0 + a1 * X[t - 1] ** 2 + b1 * sigma2[t - 1]
            #if sigma2[t] < 0:
                #print(sigma2[t])
                #print(a0)
                #print(a1)
                #print(X[t-1]**2)
                #print(b1)
                #print(sigma2[t-1])
        return sigma2

    def negative_log_likelihood(self, X, theta):

        T = len(X)

        # Estimate initial sigma squared
        initial_sigma = np.sqrt(np.mean(X ** 2))

        # Generate the squared sigma values
        sigma2 = self.compute_squared_sigmas(X, initial_sigma, theta)
        # Now actually compute
        #return -np.prod([1/sigma2[t] * np.exp(-X[t]**2/(2*sigma2[t])) for t in range(T)]# )
        return 1/T*sum([(X[t] ** 2) / (sigma2[t]) + np.log(sigma2[t]) for t in range(T)])

    def fitting(self, X):
        # Make our objective function by plugging X into our log likelihood function
        objective = partial(self.negative_log_likelihood, X)

        def constraint1(theta):
            return np.array([1 - (theta[1] + theta[2])])

        cons = ({'type': 'ineq', 'fun': constraint1})
        bnds = ((0.0, 1e12), (0.0, 1e12), (0.0, 1e12))

        result = scipy.optimize.minimize(objective, (0.1, 0.2, 0.4),
                                         method='SLSQP',
                                         constraints=cons,
                                         bounds=bnds)
        #result = scipy.optimize.minimize(objective, (0.1, 0.2, 0.4),
        #                                 method='SLSQP',
        #                                 bounds=bnds)
        theta_mle = result.x
        return theta_mle

#config = load_config(26)
#test = Garch(config)

#for i in range(0, 20000):
#    batch = test.simulate()
#    x = batch["X"]
#    y = batch["y"]
#    sigma = batch["sigma"]
    #print('mean is')
    #print(np.mean(x))
    #print('sigma is')
    #print(np.mean(sigma))
