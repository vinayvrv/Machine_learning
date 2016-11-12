from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import copy
class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
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
    def __init__( self, parameters={} ):
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
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        # self.params = {'usecolumnones': True}
        self.params = parameters
       # self.class_stats = ""
        #self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        #print Xtrain
        class_label = self.labels(Xtrain,ytrain)
        self.class_stats = self.parameter(class_label)
        #self.calprobabilities(class_stats, Xt)

    #def reset(self, parameters):
        #self.resetparams(parameters)
        # TODO: set up required variables for learning

    # TODO: implement learn and predict functions 
    def labels(self,Xtrain,ytrain):
        class_label={}
        for i in range(len(ytrain)):
            if ytrain[i] not in class_label:
                class_label[ytrain[i]]=[Xtrain[i]]
            else:
                class_label[ytrain[i]].append(Xtrain[i])
        return class_label


    def parameter(self,data):
        class_stats={}
        for key,val in data.items():
            class_stats[key]=[(utils.mean(col), utils.stdev(col)) for col in zip(*val)]
        return class_stats


    def calprobabilities(self,class_stats_d, input_data):
        associated_class=[]
        possibility = {}
        for keys, values in class_stats_d.items():
            possibility[keys] = 1
            for i in range(len(values)):
                mean, stdev = values[i]
                x=input_data[i]
                #print "value of x",x
                possibility[keys] *= utils.calculateprob(x, mean, stdev)
        #print possibility
        #input()
        val_old=float('-inf')
        for kk,vv in possibility.items():
            if vv>val_old:
                val_old=vv
                class_lab=kk
        return class_lab


    def predict(self,test_data):
        #print test_data
        predictionsof_class=[]
        #group=labels(train_data)
        #class_statistics=parameter(group)
        for i in range (len(test_data)):
            prob=self.calprobabilities(self.class_stats,test_data[i])
            predictionsof_class.append(prob)
        return predictionsof_class









class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        # self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.params = parameters
        #self.reset(parameters)

    # def reset(self, parameters):
    #     self.resetparams(parameters)
    #     self.weights = None
    #     if self.params['regularizer'] is 'l1':
    #         self.regularizer = (utils.l1, utils.dl1)
    #     elif self.params['regularizer'] is 'l2':
    #         self.regularizer = (utils.l2, utils.dl2)
    #     else:
    #         self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    # TODO: implement learn and predict functions
	
	
    def learn(self,Xtrain,ytrain):
        Xtrain = np.insert(Xtrain,0,1,axis=1)
        #self.weights= np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        self.weights = np.zeros(Xtrain.shape[1])
        #p=range(Xtrain.shape[0])
        episolon=0.0001       # 0.001 gives best results
        step=0.001
        #pred= np.dot(Xtrain,self.weights)
		#self.predicts= utils.sigmoid(pred)
		#self.threshold_p=utils.threshold_probs(self.predicts)
        count=0
        convergence=False
        while (convergence==False or  count<100):#convergence==False:# and count<20):
            oldweights=copy.deepcopy(self.weights)
        #(count <(100) and  math.sq(np.sum((oldweights-newweights)**2))<episolon):
            score= np.dot(Xtrain,self.weights)
            pred= utils.sigmoid(score)
            derivative=	np.dot(np.transpose(Xtrain),np.subtract(ytrain,pred))
            self.weights+=step*derivative
            newweights=copy.deepcopy(self.weights)
            count+=1
            diff=np.subtract(newweights,oldweights)
            sqweiths= np.sum(np.power(diff,2))
            if (np.sqrt(sqweiths)) < episolon:
                convergence=True
        return self.weights

    def predict(self,test_data):
        test_data = np.insert(test_data, 0, 1, axis=1)
        scores=np.dot(test_data,self.weights)
        predicts= utils.sigmoid(scores)
        threshold_p=utils.threshold_probs(predicts)
        return threshold_p
		
		
		
		
		
		
		
		
		
		
		# #for i in range (100) math.sq(np.sum((oldweights-newweights)**2))<episolon:
			# pred= np.dot(Xtrain,self.weights)
			# smallP= utils.sigmoid(pred)
			# bigP=np.diag(predicts)
			# id_matrix=np.identity(len(P))
			# hessian_matrix=np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,bigP),np.subtract(id_matrix,bigP)), Xtrain))
			# update=np.dot(Xtrain.T, np.subtract(ytrain, smallP))
			# self.weights+=  np.dot(hessian_matrix,update)
		# return self.weights
		
	# def errchk(Xtrain,ytrain,self.predicts)

		# for i in range (100):
			
			# pred= np.dot(Xtrain,self.weights)
			# predicts= utils.sigmoid(pred)
			# P=np.diag(predicts)
			# IdentityMat=np.identity(len(P))
			# Hessian=np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,P),np.subtract(IdentityMat,P)), Xtrain))
			# Gradient=np.dot(Xtrain.T, np.subtract(ytrain, p))
			# self.weights+=  np.dot(Hessian,Gradient)
			# error= threshold_p- ytrain
			# deri=np.dot(Xtrain,(ytrain-self.predicts))
			# self.weights+=self.weights*step*deri
			# newll=maxiumll(Xtrain,ytrain,self.weights)
			# if newll>ll:
				# ll=newll
			# else:
				# break
				
						# ll=float("-inf")
		# threshold_p=utils.threshold_probs(self.predicts)
		# P=np.diag(self.predicts);
		# IdentityMat=np.identity(len(P))
			
	# def maxiumll(Xtrain,ytrain,self.weights)
			# scores=np.dot(Xtrain,self.weights)
			# predicts= utils.sigmoid(scores)
			# logexp = np.log(predicts))
			# lp = np.sum((ytrain-1)*scores - logexp)
			# # return lp
	
	# def predict(self,test_data):
        # #print test_data
		# #predictionsof_class=[]
        # #group=labels(train_data)
        # #class_statistics=parameter(group)
    
		# scores=np.dot(test_data,self.weights)
		# predicts= utils.sigmoid(scores)
		# threshold_p=utils.threshold_probs(predicts)
		# #predictionsof_class.append(threshold_p)
        # return threshold_p

			
# class LogitReg2(Classifier):
#
#     def __init__( self, parameters={} ):
#         # Default: no regularization
#         # self.params = {'regwgt': 0.0, 'regularizer': 'None'}
# 		self.params = parameters
#         #self.reset(parameters)
#
#     def reset(self, parameters):
#         self.resetparams(parameters)
#         self.weights = None
#         if self.params['regularizer'] is 'l1':
#             self.regularizer = (utils.l1, utils.dl1)
#         elif self.params['regularizer'] is 'l2':
#             self.regularizer = (utils.l2, utils.dl2)
#         else:
#             self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
#
#     # TODO: implement learn and predict functions
#
#
# 	def learn(self,Xtrain,ytrain):
# 		Xtrain = np.insert(Xtrain, 0, 1, axis=1)
# 		self.weights= np.zeros(Xtrain.shape[1])
#
# 		#p=range(Xtrain.shape[0])
# 		episolon=0.0001
#
# 		#pred= np.dot(Xtrain,self.weights)
# 		#self.predicts= utils.sigmoid(pred)
# 		#self.threshold_p=utils.threshold_probs(self.predicts)
# 		count=0
# 		convergence=False
# 		while (convergence==False and count<100):
# 			oldweights=copy.deepcopy(self.weights)
# 		#(count <(100) and  math.sq(np.sum((oldweights-newweights)**2))<episolon):
# 			score= np.dot(Xtrain,self.weights)
# 			pred= utils.sigmoid(score)
# 			#error=ytrain-pred
# 			derivative=	np.dot(Xtrain.transpose,np.subtract(ytrain-pred))-lambda*np.sum((self.weights)**2)
# 			self.weights+=step*derivative
# 			newweights=self.weights
# 			count+=1
# 			if abs(np.sum(np.subtract(newweights-oldweights)))<episolon:
# 				convergence=True
# 		return self.weights
#
# 	def predict(self,test_data):
# 		scores=np.dot(test_data,self.weights)
# 		predicts= utils.sigmoid(scores)
# 		threshold_p=utils.threshold_probs(predicts)
#         return threshold_p
#
class LogitReg2(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        # self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.params = parameters
    #     self.reset(parameters)
    #
    # def reset(self, parameters):
    #     self.resetparams(parameters)
    #     self.weights = None
    #     if self.params['regularizer'] is 'l1':
    #         self.regularizer = (utils.l1, utils.dl1)
    #     elif self.params['regularizer'] is 'l2':
    #         self.regularizer = (utils.l2, utils.dl2)
    #     else:
    #         self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
    #
    # TODO: implement learn and predict functions


    def learn(self,Xtrain,ytrain):
        Xtrain = np.insert(Xtrain, 0, 1, axis=1)
        self.weights= np.zeros(Xtrain.shape[1])#np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        episolon=0.0001

        count=0
        convergence=False
        l2penalty=0.01
        step=0.001
        while (convergence==False or count<100):
            oldweights=copy.deepcopy(self.weights)
            score= np.dot(Xtrain,self.weights)
            pred= utils.sigmoid(score)
            update = l2penalty * self.weights
            likehod=np.dot(Xtrain.T, np.subtract(ytrain, pred))
            derivative = np.subtract(likehod,update)
            self.weights+=step*derivative
            newweights=copy.deepcopy(self.weights)
            count+=1
            diff=newweights-oldweights
            sqweiths= np.sum(np.power(diff,2))
            #print np.sqrt(sqweiths),episolon,count
            if (np.sqrt(sqweiths)) < episolon:
                convergence=True
        return self.weights


    def predict(self,test_data):
        test_data = np.insert(test_data, 0, 1, axis=1)
        scores=np.dot(test_data,self.weights)
        predicts= utils.sigmoid(scores)
        #ym = utils.mean(predicts)
        #predicts[predicts >= ym] = 1
        #predicts[predicts < ym] = 0
        threshold_p=utils.threshold_probs(predicts)
        return threshold_p
#
class LogitReg1(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        # self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.params = parameters
        #self.reset(parameters)

    # def reset(self, parameters):
    #     self.resetparams(parameters)
    #     self.weights = None
    #     if self.params['regularizer'] is 'l1':
    #         self.regularizer = (utils.l1, utils.dl1)
    #     elif self.params['regularizer'] is 'l2':
    #         self.regularizer = (utils.l2, utils.dl2)
    #     else:
    #         self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    # TODO: implement learn and predict functions


    def learn(self,Xtrain,ytrain):
        Xtrain = np.insert(Xtrain, 0, 1, axis=1)
        self.weights= np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        episolon=0.0001
        l1penalty=0.01
        count=0
        convergence=False
        while (convergence==False and count<20):
            oldweights=copy.deepcopy(self.weights)
            score= np.dot(Xtrain,self.weights)
            pred= utils.sigmoid(score)
            update = l1penalty * np.sign(self.weights)
            likehod = np.dot(Xtrain.T, np.subtract(ytrain, pred))
            derivative = np.subtract(likehod, update)
            #derivative=	np.dot(Xtrain.transpose,np.subtract(ytrain-pred))-lambda*(np.sign(self.weights))
            self.weights+=step*derivative
            newweights=copy.deepcopy(self.weights)
            count+=1
            diff=np.subtract(newweights,oldweights)
            sqweiths= np.sum(np.power(diff,2))
            if  (np.sqrt(sqweiths)) < episolon:
                convergence=True
            return self.weights


    def predict(self,test_data):
        scores=np.dot(test_data,self.weights)
        predicts= utils.sigmoid(scores)
        threshold_p=utils.threshold_probs(predicts)
        return threshold_p


#
#
#
#
#
# class NeuralNet(Classifier):
#
#     def __init__(self, parameters={}):
#         self.params = {'nh': 4,
#                         'transfer': 'sigmoid',
#                         'stepsize': 0.01,
#                         'epochs': 10}
#         self.reset(parameters)
#
#     def reset(self, parameters):
#         self.resetparams(parameters)
#         if self.params['transfer'] is 'sigmoid':
#             self.transfer = utils.sigmoid
#             self.dtransfer = utils.dsigmoid
#         else:
#             # For now, only allowing sigmoid transfer
#             raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
#         self.wi = None
#         self.wo = None
#
#     # TODO: implement learn and predict functions
#
#
#     def _evaluate(self, inputs):
#         """
#         Returns the output of the current neural network for the given input
#         The underscore indicates that this is a private function to the class NeuralNet
#         """
#         if inputs.shape[0] != self.ni:
#             raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
#
#         # hidden activations
#         ah = self.transfer(np.dot(self.wi,inputs))
#
#         # output activations
#         ao = self.transfer(np.dot(self.wo,ah))
#
#         return (ah, ao)
#
#
# class LogitRegAlternative(Classifier):
#
#     def __init__( self, parameters={} ):
#         self.reset(parameters)
#
#     def reset(self, parameters):
#         self.resetparams(parameters)
#         self.weights = None
#
#     # TODO: implement learn and predict functions
#
#
class LogitRegAlternative(Classifier):
    def __init__( self, parameters={} ):
        # Default: no regularization
        # self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.params = parameters
        #self.reset(parameters)
        #
        # def reset(self, parameters):
        # self.resetparams(parameters)
        # self.weights = None
        # if self.params['regularizer'] is 'l1':
        #     self.regularizer = (utils.l1, utils.dl1)
        # elif self.params['regularizer'] is 'l2':
        #     self.regularizer = (utils.l2, utils.dl2)
        # else:
        #     self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    # TODO: implement learn and predict functions


    def learn(self,Xtrain,ytrain):
        Xtrain = np.insert(Xtrain, 0, 1, axis=1)
        self.weights= np.zeros(Xtrain.shape[1])
        episolon=0.00001
        l1penalty=0.01
        l2penalty=0.01
        step=0.001
        count=0
        convergence=False
        while(convergence==False): #and count<20000):
            oldweights=copy.deepcopy(self.weights)
            score= np.dot(Xtrain,self.weights)
            pred= utils.sigmoid(score)
            likehod = np.dot(Xtrain.T, np.subtract(ytrain, pred))
            updatel2=l2penalty*self.weights
            likehod = np.subtract(likehod, updatel2)
            updatel1 = l1penalty * np.sign(self.weights)
            derivative = np.subtract(likehod, updatel1)
            #updatel2=l1penalty*np.sign(self.weights)
            #update=np.add(updatel1,updatel2)
            #derivative=	np.add(likehod,update)
            self.weights+=step*derivative
            newweights=copy.deepcopy(self.weights)
            count+=1
            diff = np.subtract(newweights, oldweights)
            sqweiths= np.sum(np.power(diff,2))
            print sqweiths,episolon,count
            if  sqweiths < episolon:
                convergence=True
        return self.weights

    def predict(self,test_data):
        test_data = np.insert(test_data, 0, 1, axis=1)
        scores=np.dot(test_data,self.weights)
        predicts= utils.sigmoid(scores)
        threshold_p=utils.threshold_probs(predicts)
        return threshold_p
