from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import copy

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

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

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
        # print Xtrain
        if self.params['usecolumnones']==True:
            Xtrain1=Xtrain.copy()
            class_label = self.labels(Xtrain1, ytrain)
            self.class_stats = self.parameter(class_label)
        else:
            Xtrain1 = np.delete(Xtrain,-1,axis=1)
            class_label = self.labels(Xtrain1, ytrain)
            self.class_stats = self.parameter(class_label)


    def reset(self, parameters):
        self.resetparams(parameters)

    def labels(self, Xtrain, ytrain):
        class_label = {}
        for i in range(len(ytrain)):
            if ytrain[i] not in class_label:
                class_label[ytrain[i]] = [Xtrain[i]]
            else:
                class_label[ytrain[i]].append(Xtrain[i])
        self.classlabel0=len(class_label[0])/(len(class_label[0])+len(class_label[1])) # calculating the priors
        self.classlabel1=len(class_label[1])/(len(class_label[0])+len(class_label[1])) # calculating the priors
        return class_label

    def parameter(self, data):
        class_stats = {}
        for key, val in data.items():
            class_stats[key] = [(utils.mean(col), utils.stdev(col)) for col in zip(*val)]
        return class_stats

    def calprobabilities(self, class_stats_d, input_data):
        #associated_class = []
        possibility = {}
        for keys, values in class_stats_d.items():
            possibility[keys] = 1
            for i in range(len(values)):
                mean, stdev = values[i]
                x = input_data[i]
                possibility[keys] *= utils.calculateprob(x, mean, stdev)

        val_old = float('-inf')

        for kk, vv in possibility.items():
            if kk==0:
                vv=self.classlabel0*vv # making use of prior
            elif kk==1:
                vv = self.classlabel1 * vv # making use of prior

            if vv > val_old:
                val_old = vv
                class_lab = kk
        return class_lab


    def predict(self, test_data):
        if self.params['usecolumnones']==True:
            test_data1 = test_data # making a check to remove the column of one
        else:
            test_data1 = np.delete(test_data,-1,axis=1)# making a check to remove the column of one

        predictionsof_class = []
        for i in range(len(test_data1)):
            prob = self.calprobabilities(self.class_stats, test_data1[i])
            predictionsof_class.append(prob)
        return predictionsof_class


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


            # TODO: implement learn and predict functions
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
            while (convergence == False and count < 2000):
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
            episolon = 0.000001
            count = 0
            convergence = False
            while (convergence == False and count < 2000):
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
            while (convergence == False and count < 2000):
                oldweights = copy.deepcopy(self.weights)
                episolon = 0.000001
                score = np.dot(Xtrain, self.weights)
                pred = utils.sigmoid(score)
                derivative = np.dot(np.transpose(Xtrain), np.subtract(ytrain, pred))
                self.weights += step * derivative
                newweights = copy.deepcopy(self.weights)
                count += 1
                diff = np.subtract(newweights, oldweights)
                sqweiths = np.sum(np.power(diff, 2))
                if (np.sqrt(sqweiths)) < episolon:
                    convergence = True
            return self.weights



    def predict(self, test_data):
        scores = np.dot(test_data, self.weights)
        predicts = utils.sigmoid(scores)
        threshold_p = utils.threshold_probs(predicts)
        return threshold_p


class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                       'transfer': 'sigmoid',
                       'stepsize': 0.01,
                       'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
            self.ni = 9 # we know that there are nine features and we need to change this manually
            self.no = 1 # we have one output
            self.nh = self.params['nh']
            self.step = self.params['stepsize']
            self.params['epochs']=self.params['epochs']
            self.epochs=self.params['epochs']

        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
            self.wi = None
            self.wo = None

        # renaming for tracking purpose
        node_number=copy.deepcopy(self.nh)
        node_input=copy.deepcopy(self.ni)
        node_output=copy.deepcopy(self.no)
        self.wi = 5*np.random.random_sample((node_number, node_input))-4 # generating random weights and trying to make the values with mean 0
        self.wo = 5*np.random.random_sample((node_output,node_number))-4 # generating random weights and trying to make the values with mean 0

    def learn(self, Xtrain, ytrain):
        for echo in range(self.epochs):
            for each in range(Xtrain.shape[0]):
                self.feedback(Xtrain[each], ytrain[each]) # passing single value to determine delta/change in weights

    def evaluate(self, inputs):
        """
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """

        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        # hidden activations
        ah = self.transfer(np.dot(self.wi, inputs))

        # output activations
        ao = self.transfer(np.dot(self.wo, ah))
        return (ah, ao)

    def feedback(self, Xtrain, ytrain):
        ah, values = self.evaluate(Xtrain)
        Xtrain = np.reshape(Xtrain.T, (1, Xtrain.shape[0]))
        numerator=(-(ytrain / values) + (1 - ytrain) / (1 - values)) #dividing the update in two parts was getting confused with brackets
        deniminitor=(values * (1 - values))
        change=numerator *deniminitor
        score = np.dot(self.wi, Xtrain.T)
        transfer_score = self.transfer(score)
        upd1 = change * transfer_score.T
        dtransfer_score = self.dtransfer(score)
        upd2 = change * np.multiply(self.wo.T, dtransfer_score)
        self.wo = self.wo - self.step * upd1 # Updating weights
        self.wi = self.wi - self.step * upd2 # Updating weights


    def predict(self, Xtest):
        result=[]
        value = self.transfer(np.dot(self.wi, Xtest.T))
        value=np.insert(value, -1, 1, axis=1)# creating appropriate dimensions
        #print "this is ", value
        possibility = self.transfer(np.dot(self.wo, value))
        for i in possibility:
            for j in i:
                if j<0.5: # This is based on the threshold_probs function in the utils
                    result.append([0])
                else:
                    result.append([1])
        return result





class LogitRegAlternative(Classifier):
    def __init__(self, parameters={}):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

        # TODO: implement learn and predict functions


    def learn(self, Xtrain, ytrain):

        self.weights = np.zeros(Xtrain.shape[1])
        episolon = 0.00001
        l1penalty = 0.02 # initializing L1 penalty as there are not mentioned in the script_classify
        l2penalty = 0.02# # initializing L2 penalty as there are not mentioned in the script_classify
        step = 0.001
        count = 0
        convergence = False
        while (convergence == False):  # and count<20000):
            oldweights = copy.deepcopy(self.weights)
            score = np.dot(Xtrain, self.weights)
            pred = utils.sigmoid(score)
            likehod = np.dot(Xtrain.T, np.subtract(ytrain, pred))
            updatel2 = l2penalty * self.weights
            likehod = np.subtract(likehod, updatel2)
            updatel1 = l1penalty * np.sign(self.weights)
            derivative = np.subtract(likehod, updatel1)
            # updatel2=l1penalty*np.sign(self.weights)
            # update=np.add(updatel1,updatel2)
            # derivative=	np.add(likehod,update)
            self.weights += step * derivative # calculating the derivative
            newweights = copy.deepcopy(self.weights)
            count += 1
            diff = np.subtract(newweights, oldweights)
            sqweiths = np.sum(np.power(diff, 2))
            # print sqweiths,episolon,count
            if sqweiths < episolon:
                convergence = True
        return self.weights

    def predict(self, test_data):
        scores = np.dot(test_data, self.weights)
        predicts = utils.sigmoid(scores)
        threshold_p = utils.threshold_probs(predicts)
        return threshold_p
