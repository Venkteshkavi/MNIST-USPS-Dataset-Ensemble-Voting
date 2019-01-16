
def mnist_neural_network():
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns
    
    
    #print(' **** PREPROCESSING MNIST DATASET **** ')
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 
    fit_mnist = model.fit(x_train, y_train, epochs=5)
    plt.plot(fit_mnist.history['loss'],color='green',linewidth=2)
    plt.show()
    model.evaluate(x_test, y_test)    
    preds_1 = model.predict(x_test)
    new_1 = np.zeros((len(preds_1)))
    for i in range(len(preds_1)):
        new_1[i]=np.argmax(preds_1[i])
    new_1.shape
    acc_1 = accuracy_score(y_test,new_1)
    print("accuracy MNIST datastet:",acc_1*100)
    print('')
    b = confusion_matrix(y_test,new_1)
    print(b)
    #plt.figure(figsize = (10,10))
    #sns.heatmap(b, annot=True)
    
    
    print('')
    #print(' **** Preprocessing USPS Dataset **** ')
    from PIL import Image
    import os
    import numpy as np
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
    
    
    arr_mat = np.asarray(USPSMat)
    matrixvalues = np.reshape(arr_mat,(19999,28,28))
    print('')
    print(' **** EVALUATING USPS DATASET **** ')
    model.evaluate(matrixvalues, USPSTar)
    preds = model.predict(matrixvalues)
    new = np.zeros((len(preds)))
    for i in range(len(preds)):
        new[i]=np.argmax(preds[i])
    usps_tar = np.asarray(USPSTar)
    acc = accuracy_score(usps_tar, new)
    print("accuracy USPS dataste:",acc*100) #accuracy usps data
    print('')
    a = confusion_matrix(usps_tar,new)
    print(a)
    
    
    
    
