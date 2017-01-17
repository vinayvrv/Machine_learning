import numpy as np
from sklearn import cross_validation


file = open("C:/Third Sem/Artifical Intelligence/Assignment5/andnpatt-riteagar-vrvernek-a5-master/train-data.txt")
data= file.readlines()

xtrain=[]
ytrain=[]
ytrain_id=[]

for i in data:
    i=i.strip()
    j=i.split()
    ytrain_id.append(j[0]) # photo id
    ytrain.append(int(j[1]))# label
    row = []
    for k in j[2:]:
        if k !="\n":
            row.append(k)
    xtrain.append(row)# features
    #print xtrain
X_train=np.array(xtrain)
X_train=X_train.astype(int)
y_train=np.array(ytrain)
y_train_id=np.array(ytrain_id)

xtest=[]
ytest=[]
xtest_id=[]

file = open("C:/Third Sem/Artifical Intelligence/Assignment5/andnpatt-riteagar-vrvernek-a5-master/test-data.txt")
data= file.readlines()

for i in data:
    i = i.strip()
    j=i.split()
    xtest_id.append(j[0])# photo id
    ytest.append(int(j[1]))# label
    row=[]
    for k in j[2:]:
        if k !='\n':
            row.append(k)
    xtest.append(row)# features

X_test=np.array(xtest)
X_test=X_test.astype(int)
y_test=np.array(ytest)
X_test_id=np.array(xtest_id)

print len(X_train), X_train.shape
print len(y_train), y_train.shape
print len(X_test), X_test.shape
print len(y_test), y_test.shape
# print X_test
# print y_test


#Xtrain,ytrain,X_test,y_test = cross_validation.train_test_split(xtrain, ytrain, test_size=0.4)




#data = np.loadtxt("C:/Third Sem/Artifical Intelligence/Assignment5/andnpatt-riteagar-vrvernek-a5-master/test-data.txt",dtype=np.str)




