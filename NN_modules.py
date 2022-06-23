# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:47:27 2022

@author: KTong
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization,LSTM,Bidirectional,Embedding,Masking
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,classification_report


class NeuralNetworkModel():
    def __init__(self):
        pass
    
    def two_layer_model(self,X,Y,l1_nodenum,l2_nodenum,drop_rate,activ):
        '''
        Generate a Neural Network with 2 hidden layers.

        Parameters
        ----------
        X : ndarray
            Train dataset.
        Y : ndarray
            Target column.
        l1_nodenum : int
            Number of nodes in the first hidden layer.
        l2_nodenum : int
            Number of notes in the second layer.
        drop_rate : float
            Drop rate of both hidden layers.

        Returns
        -------
        model : model
        Neural Network with 2 hidden layer.

        '''
        model=Sequential() # to create container
        model.add(Input(shape=np.shape(X)[1:])) # to add input layer
        model.add(Dense(l1_nodenum,activation='relu',name='HiddenLayer1')) # hidden layer 1
        # add neuron nodes in multiples of 2
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(l2_nodenum,activation='relu',name='HiddenLayer2')) # hidden layer 2
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(np.shape(Y)[1],activation=activ,name='OutputLayer')) # output layer
        
        model.summary()
        
        return model


    def eval_plot(self,hist):
        '''
        Generate graphs to evaluate model

        Parameters
        ----------
        hist : model
            Model fitted with train and test dataset.

        Returns
        -------
        Returns plots of loss and metrics assigned in model.compile()

        '''
        temp=[]
        
        for i in hist.history.keys():
            temp.append(i)

        for i in temp:
            if 'val_' in i:
                break
            else:
                plt.figure()
                plt.plot(hist.history[i])
                plt.plot(hist.history['val_'+i])
                plt.legend([i,'val_'+i])
                plt.xlabel('Epoch')
                plt.show()

    def model_eval(self,model,x_test,y_test,label):
        '''
        Generates confusion matrix and classification report based
        on predictions made by model using test dataset.

        Parameters
        ----------
        model : model
            Prediction model.
        x_test : ndarray
            Columns of test features.
        y_test : ndarray
            Target column of test dataset.
        label : list
            Confusion matrix labels.

        Returns
        -------
        Returns numeric report of model.evaluate(), 
        classification report and confusion matrix.

        '''
        result = model.evaluate(x_test,y_test)
        print(result) # loss, metrics
        
        y_pred=np.argmax(model.predict(x_test),axis=1)
        y_true=np.argmax(y_test,axis=1)
        
        cm=confusion_matrix(y_true,y_pred)
        cr=classification_report(y_true, y_pred)
        
        print(cr)
        
        disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label)
        disp.plot(cmap=plt.cm.Reds)
        plt.show()
        
    def RNN_model(self,x_train,y_train,mask,vocab,embed_dim,l2_node,droprate):
        '''
        Generates a Recurring Neural Network model with 2 Embedded and masked LSTM layers.
    
        Parameters
        ----------
        x_train : ndarray
            Train dataset.
        y_train : ndarray
            Target column.
        mask : int
            Masking value.
        vocab : TYPE
            Vocabulary size.
        embed_dim : TYPE
            Embedding dimension.
    
        Returns
        -------
        model : model
            Neural Network with 2 LSTM layer.
    
        '''
        model=Sequential()    
        model.add(Input(shape=(np.shape(x_train)[1])))
        model.add(Masking(mask_value=mask))
        model.add(Embedding(vocab,embed_dim))
        model.add(Bidirectional(LSTM(embed_dim,return_sequences=(True))))    
        model.add(Dropout(droprate))
        model.add(LSTM(l2_node))
        model.add(Dropout(droprate))
        model.add(Dense(np.shape(y_train)[1],activation='softmax'))
    
        model.summary()
        
        return model
