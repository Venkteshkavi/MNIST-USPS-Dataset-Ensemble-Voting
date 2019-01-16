#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 01:20:21 2018

@author: venktesh
"""
def svm_func():
    import numpy as np
    import gzip
    import pickle
    #import random
    from tqdm import tqdm
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from PIL import Image
    import os
    import time
    from sklearn.metrics import confusion_matrix
    image_size = 28
    features = image_size * image_size + 1
    num_class = 10
    reg_lambda = 0.001
    import matplotlib.pyplot as plt
    import pandas as pd

    #losses = []
    #start = time.time()
     #MNIST DATA PRE-PROCESSING
    print("Preprocessing MNIST DATA SET")
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
    #m_train_data = np.insert(m_train_data,0,1,axis=1)
    #classifier1 = SVC(kernel='rbf', C=2, gamma = 1,verbose =1)
    #svm_classifier=classifier1.fit(m_train_data,m_train_tar)
    #pkl_obj = open("svm_mnist.pkl",'wb')
    #pickle.dump(svm_classifier,pkl_obj)
    svm_pickle_open = open("svm_mnist_0.05.pkl",'rb')
    classifier1 = pickle.load(svm_pickle_open)
    a = classifier1.predict(m_test_data)
    conf_mat_svm_mnist = confusion_matrix(m_test_tar,a)
    print('')
    print(' **** Confusion Matrix MNIST SVM Classifier **** ')
    print(conf_mat_svm_mnist)
    accuracy = accuracy_score(m_test_tar,a)
    df_mnist = pd.DataFrame(a)
    df_mnist.to_csv("svm_mnist_labels.csv")
    print(accuracy*100)
    #print(a)    

    #USPS DATA-PREPROCESSING
    print("Preprocessing USPS DATA SET")
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
            #u_test_data = np.insert(u_test_data,0,1,axis=1)

    a_usps = classifier1.predict(u_test_data)
    conf_mat_svm_usps = confusion_matrix(u_test_tar,a_usps)
    print('')
    print(' **** Confusion Matrix USPS SVM Classifier **** ')
    print(conf_mat_svm_usps)
    accuracy_usps = accuracy_score(u_test_tar,a_usps)
    df_usps = pd.DataFrame(a_usps)
    df_usps.to_csv("svm_usps_labels.csv")
    print(accuracy_usps*100)
    #print(a_usps)
    #accuracy = classifier1.score(u_test_data,u_test_tar)
    #print(accuracy_usps)
    #end = time.time()
    #print(end-start)