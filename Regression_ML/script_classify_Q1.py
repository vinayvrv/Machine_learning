from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs
import time


def getaccuracy(ytest, predictions):
    correct = 0
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(ytest)):

        if ytest[i] == predictions[i]:
            correct += 1

    return (correct / float(len(ytest))) * 100.0


def getaccuracy1(ytest, predictions):
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytest)):
        bernulli = set(predictions)
        #print bernulli
        bernulli = list(bernulli)
        if ytest[i] == predictions[i]:
            correct += 1
        if ytest[i] == 1.0:
            if ytest[i] == predictions[i]:
                TP += 1
            else:
                FN += 1
        if ytest[i] == 0.0:
            if ytest[i] == predictions[i]:
                TN += 1
            else:
                FP += 1
    print TP
    print TN
    print FP
    print FN
    print "done"

    return (correct / float(len(ytest))) * 100.0



def geterror(ytest, predictions):
    return (100.0 - getaccuracy(ytest, predictions))


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 15



    classalgs = {#'Random': algs.Classifier(),

                 'Logistic Regression': algs.LogitReg(),

                'RBFLogitReg': algs.RBFLogitReg({'k': 50, 's': 0.2})


                 }
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4,'k': 50, 's': 1.8,},
        {'regwgt': 0.01, 'nh': 8,'k': 50, 's': 0.5,},
        {'regwgt': 0.05, 'nh': 16,'k': 50, 's': 1.0,},
    )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:

        errors[learnername] = np.zeros((numparams, numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize, testsize)

        trainset1, testset1 = dtl.load_susy_complete(trainsize,testsize)

        print('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],
                                                                              r)

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                # Reset learner for new parameters
                if learnername  in ('Logistic Regression',"RBFLogitReg", 'L2 Logistic Regression', 'L1 Logistic Regression'): # creating check for which data set to run the alogs
                    learner.reset(params)
                    print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
                # Train model
                    st=time.time()
                    learner.learn(trainset[0], trainset[1])
                # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    print 'Error for ' + learnername + ': ' + str(error)
                    errors[learnername][p, r] = error
                elif learnername not in ('Logistic Alternative','RBFLogitReg','Question 2', 'L2 Logistic Regression', 'L1 Logistic Regression'):
                    learner.reset(params)
                    print "second"
                    print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
                    # Train model
                    st = time.time()
                    #learner.learn(trainset1[0], trainset1[1])
                    # Test model
                    predictions = learner.predict(testset1[0])
                    error = geterror(testset1[1], predictions)
                    print 'Error for ' + learnername + ': ' + str(error)
                    errors[learnername][p, r] = error

    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print 'Best parameters for ' + learnername + ': ' + str(learner.getparams())
        print 'Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(
            1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns))
  
