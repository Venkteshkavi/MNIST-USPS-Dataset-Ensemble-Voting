#!/usr/bin/env python
# coding: utf-8

# In[4]:

def ensemble_voting_func():
    import pandas as pd
    import numpy as np
    import math
    import os
    import gzip
    import pickle
    from PIL import Image
    label_list = []
    usps_label_list = []
    
    
    # In[5]:
    
    
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
    
    
    # In[6]:
    
    
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
    
    
    # In[7]:
    
    
    m_test_tar
    
    
    # In[8]:
    
    
    logistic_mnist_labels = pd.read_csv("logistic_mnist_labels.csv")
    logistic_usps_labels = pd.read_csv("logistic_usps_labels.csv")
    dneural_mnist_labels = pd.read_csv("dneural_mnist_labels.csv")
    dneural_usps_labels = pd.read_csv("dneural_usps_labels.csv")
    svm_mnist_labels = pd.read_csv("svm_mnist_labels.csv")
    svm_usps_labels = pd.read_csv("svm_usps_labels.csv")
    random_mnist_labels = pd.read_csv("randomforest_mnist_labels.csv")
    random_usps_labels = pd.read_csv("randomforest_usps_labels.csv")
    
    
    # In[9]:
    
    
    logistic_mnist_labels.drop(logistic_mnist_labels.columns[0:1],axis=1,inplace=True)
    logistic_usps_labels.drop(logistic_usps_labels.columns[0:1],axis=1,inplace=True)
    dneural_mnist_labels.drop(dneural_mnist_labels.columns[0:1],axis=1,inplace=True)
    dneural_usps_labels.drop(dneural_usps_labels.columns[0:1],axis=1,inplace=True)
    svm_mnist_labels.drop(svm_mnist_labels.columns[0:1],axis=1,inplace=True)
    svm_usps_labels.drop(svm_usps_labels.columns[0:1],axis=1,inplace=True)
    random_mnist_labels.drop(random_mnist_labels.columns[0:1],axis=1,inplace=True)
    random_usps_labels.drop(random_usps_labels.columns[0:1],axis=1,inplace=True)
    
    
    # In[10]:
    
    
    logistic_mnist_labels=logistic_mnist_labels.T
    logistic_usps_labels = logistic_usps_labels.T
    dneural_mnist_labels = dneural_mnist_labels.T
    dneural_usps_labels = dneural_usps_labels.T
    svm_mnist_labels = svm_mnist_labels.T
    svm_usps_labels = svm_usps_labels.T
    random_mnist_labels = random_mnist_labels.T
    random_usps_labels = random_usps_labels.T
    
    
    # In[11]:
    
    
    #logistic_mnist_labels
    
    
    # In[12]:
    
    
    #svm_mnist_labels
    
    
    # In[13]:
    
    
    #random_mnist_labels
    
    
    # In[14]:
    
    
    #dneural_mnist_labels
    
    
    # In[15]:
    
    
    frames = [logistic_mnist_labels,svm_mnist_labels,random_mnist_labels,dneural_mnist_labels]
    mnist_labels_matrix = pd.concat(frames)
    
    
    # In[34]:
    
    
    print(' ****Predicted Labels of All Classifiers of MNIST DATA ***** ')
    print('row1 : Logistic....rows2: SVM....row3: Random Forest...rows4: Deep Neural Network')
    print('')
    print(mnist_labels_matrix)
    print('')
    
    
    # In[17]:
    
    
    for i in range(mnist_labels_matrix.shape[1]):
        label_list.append(mnist_labels_matrix[i].mode())
        i = i+1
    
    
    # In[18]:
    
    
    final_label_matrix = pd.DataFrame(label_list)
    
    
    # In[19]:
    
    
    final_label_matrix.drop(final_label_matrix.columns[[1,2,3]],axis=1,inplace=True)
    
    
    # In[20]:
    
    
    final_label_matrix=final_label_matrix.T
    
    
    # In[21]:
    
    
    final_label_matrix_np = final_label_matrix.values
    final_label_matrix_np = np.asarray(final_label_matrix_np)
    #final_label_matrix_np = final_label_matrix.T
    
    
    # In[22]:
    
    
    final_label_matrix_np
    
    
    # ### CALCULATING ACCURACY
    
    # In[23]:
    
    
    summed_values = np.sum(np.equal(final_label_matrix_np,m_test_tar))
    #print(summed_values)
    accuracy_count = summed_values.sum(axis=0)
    #print(accuracy_count)
    
    
    # In[24]:
    
    
    accuracy = (accuracy_count / 10000) * 100
    print(' **** ENSEMBLE VOTING FOR MNIST DATA ACCURACY**** ')
    print(accuracy,"%")
    print('')
    
    
    # In[25]:
    
    
    frames_usps = [logistic_usps_labels,svm_usps_labels,random_usps_labels,dneural_usps_labels]
    usps_labels_matrix = pd.concat(frames_usps)
    usps_labels_matrix.shape[1]
    print(' ****Predicted Labels of All Classifiers of USPS DATA ***** ')
    print('')
    print(usps_labels_matrix)
    print('')
    
    
    # In[26]:
    
    
    for i in range(usps_labels_matrix.shape[1]):
        usps_label_list.append(usps_labels_matrix[i].mode())
        i = i+1
    
    
    # In[27]:
    
    
    usps_labels_matrix = pd.DataFrame(usps_label_list)
    usps_labels_matrix.drop(usps_labels_matrix.columns[[1,2,3]],axis=1,inplace=True)
    usps_labels_matrix = usps_labels_matrix.T
    usps_labels_matrix_np = usps_labels_matrix.values
    usps_labels_matrix_np = np.asarray(usps_labels_matrix_np)
    
    
    # ### CALCULATING ACCURACY OF USPS
    
    # In[35]:
    
    
    summed_values_usps = np.sum(np.equal(usps_labels_matrix_np,u_test_tar))
    accuary_count_usps = summed_values_usps.sum(axis=0)
    accuracy1 = math.ceil((accuary_count_usps/19999) * 100)
    print(' **** ENSEMBLE VOTING ACCURACY OF USPS DATASET **** ')
    print(accuracy1,'%')
    
    
    # In[ ]:
    
    
    
    
