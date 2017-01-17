from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import copy
import random
import math
from scipy.cluster.vq import kmeans

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    # def resetparams(self, parameters):
    #     """ Can pass parameters to reset with new parameters """
    #     try:
    #         utils.update_dictionary_items(self.params, parameters)
    #     except AttributeError:
    #         # Variable self.params does not exist, so not updated
    #         # Create an empty set of params for future reference
    #         self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(
            np.add(np.dot(Xtrain.T, Xtrain) / numsamples, self.params['regwgt'] * np.identity(Xtrain.shape[1]))),
                                     Xtrain.T), yt) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        # self.params = {'usecolumnones': True}
        self.params = parameters
        # self.class_stats = ""
        # self.reset(parameters)

    def learn(self, Xtrain, ytrain):

        self.classlabel1 = sum(ytrain) / len(ytrain)  # calculating the priors since i know its only one and zero
        self.classlabel0 = 1 - self.classlabel1  # calculating the priors

        if self.params['usecolumnones'] == True:
            Xtrain1 = Xtrain.copy()
            class_label = self.labels(Xtrain1, ytrain)
            self.class_stats = self.parameter(class_label)
        else:
            Xtrain1 = np.delete(Xtrain, -1, axis=1)
            class_label = self.labels(Xtrain1, ytrain)
            self.class_stats = self.parameter(class_label)

    def reset(self, parameters):
        #self.resetparams(parameters)
        pass
    def labels(self, Xtrain, ytrain):
        class_label = {}
        for i in range(len(ytrain)):
            if ytrain[i] not in class_label:
                class_label[ytrain[i]] = [Xtrain[i]]
            else:
                class_label[ytrain[i]].append(Xtrain[i])

        return class_label

    def parameter(self, data):
        class_stats = {}
        for key, val in data.items():
            class_stats[key] = [(utils.mean(col), utils.stdev(col)) for col in zip(*val)]
        return class_stats

    def calprobabilities(self, class_stats_d, input_data):
        # associated_class = []
        possibility = {}
        for keys, values in class_stats_d.items():
            possibility[keys] = 1
            for i in range(len(values)):
                mean, stdev = values[i]
                x = input_data[i]
                possibility[keys] *= utils.calculateprob(x, mean, stdev)

        val_old = float('-inf')

        for kk, vv in possibility.items():  # checking each key and value in the dictionary
            if kk == 0:
                vv = self.classlabel0 * vv  # making use of prior
            elif kk == 1:
                vv = self.classlabel1 * vv  # making use of prior

            if vv > val_old:
                val_old = vv
                class_lab = kk
        return class_lab

    def predict(self, test_data):
        if self.params['usecolumnones'] == True:
            test_data1 = test_data  # making a check to remove the column of one
        else:
            test_data1 = np.delete(test_data, -1, axis=1)  # making a check to remove the column of one

        predictionsof_class = []
        for i in range(len(test_data1)):
            prob = self.calprobabilities(self.class_stats, test_data1[i])
            predictionsof_class.append(prob)
        return predictionsof_class


######################################################
#  RBF part implementation
######################################################
class RBFgaussian(Classifier): # implementing RBF Function
    def __init__(self, parameters=None):
        self.weights = None
        if parameters is not None :
            self.params=parameters
            self.k = self.params['k']
            self.sigma = self.params['s']
            self.regwgt = 0.001
            self.centers = []
        else:
            print "my params"
            self.regwgt = 0.001
            self.k = 60
            self.sigma = 1.0
            self.centers = []


    def learn(self, Xtrain, ytrain):
        self.weights = np.zeros(self.k)
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        self.centers = []
        a = np.arange(Xtrain.shape[0])
        np.random.shuffle(a)
        local=a[:self.k]
        for i in  local:
            self.centers.append(Xtrain[i])
        self.centers=np.array(self.centers)

        phi = []
        for x in (Xtrain):
            phirow=[]
            for c in (self.centers):
                dist = np.linalg.norm(x -c)*np.linalg.norm(x -c)
                value = math.exp(-self.sigma* dist)
                phirow.append(value)
            phi.append(phirow)

        phi = np.array(phi)

        self.weights=np.dot(np.dot(np.linalg.inv(np.dot(phi.T,phi)+np.dot(penalty,np.identity(phi.shape[1]))), phi.T),yt) #adding a litte regularization term

        return self.weights

    def predict(self, Xtest):
        test = []
        for x in (Xtest):
            newrow=[]
            for c in (self.centers):
                dist = np.linalg.norm(x -c)*np.linalg.norm(x -c)
                value = math.exp(-self.sigma* dist)  # based on the formula  exp -b|| a-C|| square
                newrow.append(value)
            test.append(newrow)

        test=np.array(test)
        ytest = np.dot(test, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest


#############################################################################################################
# RBF transformation and then using logistic regression
################################################################################################################
class RBFLogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        # if len(parameters)==0:
        #     self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        if parameters is not None:
            self.params = parameters
            self.k = parameters['k']
            self.sigma = parameters['s']
            self.centers = []
        else:
            print "Custom params choosen"
            self.regwgt = 0.001
            self.k = 50
            self.sigma = 0.5
            self.centers = []
            self.params = parameters

    def reset(self, parameters):
        self.resetparams(parameters)
        if "regularizer" in self.params:
            if self.params['regularizer'] is 'l1':
                self.regularizer = (utils.l1, utils.dl1)
            elif self.params['regularizer'] is 'l2':
                self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def resetparams(self, parameters):

        #print parameters
        self.params['k'] = parameters['k']
        self.k=self.params['k']
        self.params['s'] = parameters['s']
        self.sigma = self.params['s']

    def learn(self, Xtrain, ytrain):

        step=0.0001      #lower step size selected
        episolon = 0.00001
        convergence=False

        count = 0
        self.centers = []
        a = np.arange(Xtrain.shape[0])
        np.random.shuffle(a)
        local = a[:self.k]
        for i in local:
            self.centers.append(Xtrain[i])
        self.centers = np.array(self.centers)
        phi = []
        for x in (Xtrain): # transforming matrix
            phirow = []
            for c in (self.centers):
                dist = np.linalg.norm(x-c)*np.linalg.norm(x-c)
                value =math.exp (-self.sigma * dist)
                phirow.append(value)
            phirow.append(1)
            phi.append(phirow)
        phi = np.array(phi)
        Xtrain1=phi

        self.weights = np.zeros(Xtrain1.shape[1])


        while (convergence == False):
            oldweights = copy.deepcopy(self.weights)
            score = np.dot(Xtrain1, self.weights) # this is like regression score
            pred = utils.sigmoid(score)
            derivative = np.dot(np.transpose(Xtrain1), np.subtract(ytrain, pred))
            self.weights += step * derivative # updating weights
            newweights = copy.deepcopy(self.weights)
            count += 1
            diff = np.subtract(newweights, oldweights)
            sqweiths = np.sum(np.power(diff, 2))
            #print np.sqrt(sqweiths)
            if (np.sqrt(sqweiths)) < episolon:
                convergence = True
        return self.weights



    def predict(self, Xtest):
        test = []
        for x in (Xtest):
            newrow=[]
            for c in (self.centers):
                dist = np.linalg.norm(x-c)*np.linalg.norm(x-c)
                value = math.exp(-self.sigma* dist)
                newrow.append(value)
            newrow.append(1)
            test.append(newrow)


        test=np.array(test)
        scores = np.dot(test, self.weights)
        predicts = utils.sigmoid(scores)
        threshold_p = utils.threshold_probs(predicts)
        return threshold_p



###########################
##Logistic regression
##########################

class LogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        if len(parameters)==0:
            self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        else:
            self.params = parameters

    def reset(self, parameters):
        self.resetparams(parameters)

        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def resetparams(self, parameters):
        self.params['regwgt'] = parameters['regwgt']

    def learn(self, Xtrain, ytrain):
        self.weights = np.zeros(Xtrain.shape[1])
        step=0.001
        convergence=False
        count=0
        if self.params['regularizer'] == "l2":

            l2penalty = self.params['regwgt']
            count = 0
            episolon = 0.000001 #setting threshold
            convergence = False
            while (convergence == False and count < 5000):
                oldweights = copy.deepcopy(self.weights)
                score = np.dot(Xtrain, self.weights)
                pred = utils.sigmoid(score)
                update = l2penalty * self.weights
                likehod = np.dot(Xtrain.T, np.subtract(ytrain, pred))
                derivative = np.subtract(likehod, update)
                self.weights += step * derivative # updating weights
                newweights = copy.deepcopy(self.weights)
                count += 1
                diff = np.subtract(newweights, oldweights)
                sqweiths = np.sum(np.power(diff, 2))
                if (np.sqrt(sqweiths)) < episolon:
                    convergence = True
            return self.weights
        elif self.params['regularizer'] == "l1":

            l1penalty = self.params['regwgt']
            self.weights = np.zeros(Xtrain.shape[1])
            episolon = 0.000001
            count = 0
            convergence = False
            while (convergence == False and count < 5000):
                oldweights = copy.deepcopy(self.weights)
                score = np.dot(Xtrain, self.weights)
                pred = utils.sigmoid(score)
                update = l1penalty * np.sign(self.weights)
                likehod = np.dot(Xtrain.T, np.subtract(ytrain, pred))
                derivative = np.subtract(likehod, update)
                self.weights += step * derivative
                newweights = copy.deepcopy(self.weights)
                count += 1
                diff = np.subtract(newweights, oldweights)
                sqweiths = np.sum(np.power(diff, 2))
                if (np.sqrt(sqweiths)) < episolon:
                    convergence = True
            return self.weights
        else:
            episolon = 0.000001
            while (convergence == False and count < 5000):
                oldweights = copy.deepcopy(self.weights)
                #print episolon
                score = np.dot(Xtrain, self.weights) # this is like regrseeion score
                pred = utils.sigmoid(score)
                derivative = np.dot(np.transpose(Xtrain), np.subtract(ytrain, pred))
                self.weights += step * derivative # updating weights
                newweights = copy.deepcopy(self.weights)
                count += 1
                diff = np.subtract(newweights, oldweights)
                sqweiths = np.sum(np.power(diff, 2))
                #print np.sqrt(sqweiths)
                if (np.sqrt(sqweiths)) < episolon:
                    convergence = True
            return self.weights



    def predict(self, test_data):
        scores = np.dot(test_data, self.weights)
        predicts = utils.sigmoid(scores)
        threshold_p = utils.threshold_probs(predicts)
        return threshold_p

