#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 15 20:32:35 2018

@author: venktesh
"""

import numpy as np
import gzip
import pickle
import random
from tqdm import tqdm
from PIL import Image
import os
import tensorflow as tf
mnist = tf.keras.datasets.mnist
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from neuralnet_mnist1 import mnist_neural_network
from testing_svm import svm_func
from main_randomforest import random_forest_func
from ensemble_voting import ensemble_voting_func

image_size = 28
features = image_size * image_size + 1
num_class = 10
reg_lambda = 0.001


losses = []

# MNIST DATA PRE-PROCESSING
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()
m_train_tar = np.array(training_data[1])
m_val_tar = np.array(validation_data[1])
m_test_tar = np.array(test_data[1])
m_train_data = np.array(training_data[0])
m_val_data = np.array(validation_data[0])
m_test_data = np.array(test_data[0])
    
#Adding Bias Term to Data
m_train_data = np.insert(m_train_data,0,1,axis=1)
m_test_data = np.insert(m_test_data,0,1,axis=1)
    
#USPS DATA-PREPROCESSING
USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
u_test_data = np.array(USPSMat)
u_test_tar = np.array(USPSTar)
#ADDING BIAS TERM TO USPS DATA
u_test_data = np.insert(u_test_data,0,1,axis=1)


#LOGISTIC REGRESSION FUNCTIONS

def softmax(activation):
    vec = np.exp(activation)
    pr_res = np.zeros((activation.shape[0],activation.shape[1]))
    sum_vec = vec.sum(axis=1)
    for i in range(activation.shape[0]):
        for j in range(10):
            pr_res[i][j] = vec[i][j] / sum_vec[i]
    return pr_res

def hotvec_prep(m_train_tar):
    res = np.zeros((m_train_tar.size,num_class))
    for i in range(m_train_tar.size):
        for j in range(num_class):
            if(j == m_train_tar[i]):
                res[i][j] = 1
    return res

def get_error(predicted_label,hot_vec,m_train_data):
    hot_vec = hot_vec.reshape((50000,10))
    s = np.matmul(m_train_data.T,predicted_label - hot_vec)
    return s

def loss_calc(predicted_label,hot_vec):
    #h = np.argmax(hot_vec, axis=1)
    #y = np.argmax(predicted_label,axis=1)
    log_h = np.log(predicted_label)
    #log_h[log_h == -inf] = 0
    #nlog_h = np.nan_to_num(log_h)
    #ones = np.argmax(hot_vec,axis=1)
    #loss_pro = np.zeros(50000,1)
    #loss_pro = 
    loss_x = np.sum(hot_vec * log_h)
    loss_x = -(loss_x) / predicted_label.shape[0]    
    losses.append(loss_x)
    #print(loss_x)
    return losses
    
def logistic_reg_training():
    epochs = 1000
    learning_param = 0.9
    rows_data = m_train_data.shape[0]
    w = np.random.rand(features,num_class)
    hot_vec = hotvec_prep(m_train_tar)
    lr = learning_param/rows_data
    for i in tqdm(range(epochs)):
        activation = np.matmul(m_train_data,w)
        predicted_label = softmax(activation)
        losses = loss_calc(predicted_label,hot_vec)
        gradient = get_error(predicted_label,hot_vec,m_train_data)
        regularized_w = reg_lambda * w
        regularized_w[0,:] = 0
        w = w - lr*(gradient + regularized_w)
        i = i+1
    plt.plot(losses)
    plt.show()        
    print('')
    return w
def logistic_reg_test(w):
    testing_labels = softmax(np.matmul(u_test_data, w))
    predicted_output_testing = np.argmax(testing_labels,axis=1)
    acc1 = np.sum(np.equal(u_test_tar,predicted_output_testing))
    conf_mat1 = confusion_matrix(u_test_tar,predicted_output_testing)
    df = pd.DataFrame(predicted_output_testing)
    df.to_csv("logistic_usps_labels.csv")
    #print(predicted_output_testing)
    #print(acc1)
    print('  **** Confusion Matrix USPS Dataset Logistic Regression ****  ')
    print(conf_mat1)
    print('')
    return acc1/predicted_output_testing.shape[0]

def get_accuracy(w,m_test_data,m_test_tar):
    final_predicted_label = softmax(np.matmul(m_test_data,w))
    predicted_output = np.argmax(final_predicted_label,axis=1)
    acc = np.sum(np.equal(predicted_output,m_test_tar))
    df1 = pd.DataFrame(predicted_output)
    df1.to_csv("logistic_mnist_labels.csv")
    #print(predicted_output)
    conf_mat = confusion_matrix(m_test_tar,predicted_output)
    #print(acc)
    print('  **** Confusion Matrix MNIST Datset Logistic Regression ****  ')
    print(conf_mat)
    return acc/final_predicted_label.shape[0]

def main_func():
    print(' **** LOGISTIC REGRESSION **** ')
    updated_weights = logistic_reg_training()
    accuracy = get_accuracy(updated_weights,m_test_data,m_test_tar)
    test_acc = logistic_reg_test(updated_weights)
    print("MNIST DATASET ACCURACY :",accuracy*100,"%")
    print("USPS DATASET ACCURACY :",test_acc*100,"%")   
    print('')
    print('     *********** NEURAL NETWORK **********     ')
    mnist_neural_network()
    print('')
    print('     ***** SVM CLASSIFIER ****     ')
    svm_func()
    print('')
    #print('     ***** Random Forest *****     ')
    random_forest_func()
    print(' **** ENSEMBLE VOTING FOR ALL CLASSIFIERS **** ')
    print('')
    #ensemble_voting_func()
main_func()    
