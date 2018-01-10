from numpy import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.core.utils.data_structures import InstanceWindow
from collections import Counter
class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        # TODO
        self.max_models=max_models
        self.window_size=window_size
        #self.window=[]
        self.window=InstanceWindow(window_size)
        self.number_element=0
        #for i in range(self.max_models):
        #    self.h=DecisionTreeClassifier()
         #   self.H.append(self.h)

    def partial_fit(self, X, y=None, classes=None):
        # TODO 
        #if not initialized ...
            # Setup 
        # N.B.: The 'classes' option is not important for this classifier
        # HINT: You can build a decision tree model on a set of data like this:
        #       h = DecisionTreeClassifier()
        #       h.fit(X_batch,y_batch)
        #       self.H.append(h) # <-- and append it to the ensemble
        r,c=X.shape
        for i in range(r):
            if self.window is None:
                self.window=InstanceWindow(self.window_size)
            self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            self.number_element+=1
            if self.h==None:
                self.h=DecisionTreeClassifier()
            if self.number_element==self.window_size:
                X_batch=self.window.get_attributes_matrix()
                y_batch=self.window.get_targets_matrix()
                self.h.fit(X_batch,y_batch)
                if(len(self.H)==self.max_models):
                    self.H.pop(0)
                self.H.append(self.h)
                self.number_element=0
            
        return self

    def predict(self, X):
        # TODO 
        N,D = X.shape
        # You also need to change this line to return your prediction instead of 0s:
        Y_maj=zeros(N)
        #print("N: "+str(N))
        y_pred=zeros(len(self.H))
        for j in range(len(self.H)):
                #print("testssssss")
            y_pred[j]=self.H[j].predict(X)
            #print(y_pred[j])
                #print(self.H[j].predict(np.asarray([X[i]])))
                #print("trainingkdsljfl")
        class_counts=Counter(y_pred)
                #print("class_counts: "+class_counts)
        top_one=class_counts.most_common(1)
                #print("top_one value: "+top_one[0][0])
        Y_maj[0]=(top_one[0][0])
        #print("Y_maj: "+Y_maj)

        return Y_maj
