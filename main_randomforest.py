#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:55:34 2018

@author: venktesh
"""
def random_forest_func():
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.datasets import fetch_mldata
    from keras.datasets import mnist
    from sklearn.metrics import accuracy_score
    import os
    from PIL import Image
    import pandas as pd
    import pickle
    from tqdm import tqdm
    from sklearn.metrics import confusion_matrix
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #USPS DATA-PREPROCESSING
    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []

    for j in tqdm(range(0,10)):
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

    #mnist = fetch_mldata('MNIST original')
    #n_train = 60000
    #n_test = 10000
    #indices = arange(len(mnist.data))
    #train_idx = arange(0,n_train)
    #test_idx = arange(n_train+1,n_train+n_test)
    #X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
    #X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))
    print('Fitting in Progress')
    #RandomForestClassifier MNIST DATASET
    #classifier1 = RandomForestClassifier(n_estimators=10,n_jobs=-1,verbose=2)
    #random_forest =classifier1.fit(x_train, y_train)
    #pkl_obj = open("random_forest_mnist.pkl",'wb')
    #pickle.dump(random_forest,pkl_obj)
    pickle_open = open("random_forest_mnist.pkl",'rb')
    classifier1 = pickle.load(pickle_open)
    a = classifier1.predict(x_test)
    df_mnist = pd.DataFrame(a)
    df_mnist.to_csv("randomforest_mnist_labels.csv")
    conf_mat_random = confusion_matrix(y_test,a)
    print(' **** Confusion Matrix MNIST Random Forest **** ')
    print(conf_mat_random)
    accuracy = accuracy_score(y_test,a)
    b = classifier1.predict(u_test_data)
    df_usps = pd.DataFrame(b)
    df_usps.to_csv("randomforest_usps_labels.csv")
    conf_mat_random_usps = confusion_matrix(u_test_tar,b)
    print(' **** Confusion Matrix USPS Random Forest **** ')
    print(conf_mat_random_usps)
    accuracy2 = accuracy_score(u_test_tar,b)
    print(accuracy,accuracy2)
