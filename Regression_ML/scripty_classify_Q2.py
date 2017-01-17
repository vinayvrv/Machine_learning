from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from scipy import stats
import dataloader as dtl
import classalgorithms as algs
import time
from sklearn import cross_validation
from sklearn.cross_validation import KFold

def getaccuracy(ytest, predictions):
    correct = 0
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

# def geterror(ytest, predictions):
#     return (100.0 - getaccuracy1(ytest, predictions)) #used for confusion matrix

def geterror(ytest, predictions):
    return (100.0 - getaccuracy(ytest, predictions))



if __name__ == '__main__':
    holder=[]
    naive=[]
    lr=[]
    lr1=[]
    for ratio in [0.1, 0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #training split
        trainsize = int(30000*ratio)
        testsize = int(30000*(1-ratio))
        numruns = 25


        classalgs = {
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'L1 Logistic Regression': algs.LogitReg({'regularizer': 'l1','regwgt':0.01}),
                 'L2 Logistic Regression': algs.LogitReg({'regularizer': 'l2','regwgt':0.01})}


        for learnername, learner in classalgs.iteritems():
            counter = []
            for r in range(numruns):

                trainset, testset = dtl.load_card(trainsize, testsize)

                print('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r),
                print "on parameters" + str(learner.getparams())

                    # Train model
                learner.learn(trainset[0], trainset[1])
                    # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print "The Error for "+str(learnername)+" is "+str(error)+"\n"
                counter.append(error)
            m_error=np.mean(counter)
            st_error=np.std(counter)
            if learnername=='Naive Bayes':
                naive.append(m_error)
            elif learnername=='L2 Logistic Regression':
                lr.append(m_error)
            elif learnername=='L1 Logistic Regression':
                lr1.append(m_error)
            holder.append([learnername,m_error,st_error,ratio,abs(1-ratio)])
    infer = sorted(holder, key=lambda x: x[1])
    #print infer

    for learnername in classalgs.keys():

        for i in range(len(infer)):


            if infer[i][0]==learnername:
                print "-"*20
                print "The average error for "+str(learnername) + " is "+str(infer[i][1])+" with Std Error of "+str(infer[i][2])
                print "for split, train: "+str(infer[i][3])+ " and test: "+ str(infer[i][4])
                break

    # data = [naive,lr,lr1]
    # writer = csv.writer(open("C:\Third Sem\Machine Learning\Assigment_4\\test_figures.csv", 'w'))
    # for row in data:
    #     writer.writerow(row)

    trainsize=int(15000)# the data is already randomized using excel
    testsize=int(15000)# # the data is already randomized using excel
    trainset, testset = dtl.load_card1(trainsize, testsize) # loading the data
    train_in=np.vstack((trainset[0],testset[0]))
    train_out = list(trainset[1]) +list(testset[1])
    train_out=np.array(train_out)
    cv = cross_validation.KFold(train_in.shape[0], n_folds=5) #creating K folds


    classalgs = {
               'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                'L2 Logistic Regression': algs.LogitReg({'regularizer': 'l2','regwgt':0.0}),
               'L1 Logistic Regression': algs.LogitReg({'regularizer': 'l1','regwgt':0.0})} #selected these 3 algorithms

    main_details=[]
    main_dic={}
    for learnername, learner in classalgs.iteritems():
        alphas=np.logspace(-2, 1, num=10) # using numpy.logspace to generate meta/tuning parameter
        #alphas=[0.001,0.01]#,0.1,0.3,0.4,0.5,0.6,1.0,1.5,2.0]
        for alpha in alphas:
            learner.reset({'regwgt':alpha})
            error_list = []
            print "Running K-Fold validation"
            for train, test in cv:

                learner.learn(train_in[train],train_out[train])
                predictions = learner.predict(train_in[test])
                error = geterror(train_out[test], predictions)
                error_list.append(error)
            error_list=np.array(error_list)

            newkey= learnername+str(",")+str(alpha)
            main_dic[newkey]=error_list
            if learnername=="L1 Logistic Regression" :
                print "The mean of error for Method "+str(learnername)+ " with tuning paramater "+str(alpha)+ " is "+str(np.mean(error_list))
                print "and the Std deviation of error for Method " + str(learnername) + " is ", np.std(error_list)
                print "......"
                main_details.append([learnername,alpha,np.mean(error_list),np.std(error_list)])
            elif learnername=="L2 Logistic Regression" :
                print "The mean of error for Method "+str(learnername)+ " with tuning paramater "+str(alpha)+ " is "+str(np.mean(error_list))
                print "and the Std deviation of error for Method " + str(learnername) + " is ", np.std(error_list)
                print "......"
                main_details.append([learnername,alpha,np.mean(error_list),np.std(error_list)])
            else:
                print "The mean of error for Method " + str(learnername) + " is "+ str(np.mean(error_list))
                print "The Std deviation of error for Method " + str(learnername) + " is "+ str( np.std(error_list))
                print "......"
                main_details.append([learnername, alpha, np.mean(error_list), np.std(error_list)])

    infer=sorted(main_details,key=lambda x:x[2])


    #print infer
    test=[]  # stores Learner name
    test1=[] #stores alpha
    test3=[] #stores mean error
    tt=[]
    while len(test)<2: # as we are looking for top two algos to compare
        for i in infer:
            if i[0] not in test:
                test.append(i[0])
                test1.append(i[1])
                test3.append(i[2])
    print "Out of the three Algorithms The best two are as follows: "
    print "The Best result for is for Algorithm " +test[0]+" with Avg Error of "+str(test3[0])+ " with Tuning/penalty Parameter of "+ str(test1[0])
    print "The Second Best result for is for Algorithm " + test[1] + " with Avg Error of " + str(
        test3[1]) + " with Tuning/penalty Parameter of " + str(test1[1])
    for i in range (3): #
        mykey=test[i]+","+str(test1[i])
        tt.append((test[i],test1[i],main_dic.get(mykey)))

    alg1,tuning1,val1=tt[0]
    alg2,tuning2,val2=tt[1]

    Tvalue,Pvalue = stats.ttest_ind(val1, val2)# the t=test
    print "The t-statistic value is ",Tvalue
    print "The P- value is ",Pvalue
    significance = 0.05

    print "Null Hypothesis: mean error of these two algorithms are not statistically different "+'\n'
    print "Alternate Hypothesis:  mean error from the algorithms are statistically different "+'\n'
    if Pvalue < significance:
        print "We reject the Null Hypothesis as its P-value of "+str(Pvalue)+" is less than significance level of "+str(significance)
        print "These two alogrithms are different, as the difference in their means is not due chance"
    else:
        print "We do not reject the Null Hypothesis as " + str(Pvalue) + " is greater than significance level of " + str(significance)
        print "\n"
        print "As we accepted the Null hypothesis it is adviceable to select we can also note that both the algorithms are not statistically different, but still we choose the one with least error for a given Lambda value"





