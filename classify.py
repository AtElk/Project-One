# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:43:24 2022

@author: Elijah Gallagher

"""



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import reshape as re
import numpy as np




tf.get_logger().setLevel('WARNING')


#the error check is good for running for the first time but bloated any other time
#i wanted a way to build/train the models in a seperate program because importing the .py file is like 1 second of wasted time and
#actually training them is minutes. 
errorcheck = True
if errorcheck == True:
    fails = 0
    try:
        model_conv = tf.keras.models.load_model('conv_CIF10.model')
    except:
        print('conv model failed to load')
        fails += 1
    try:
        model_overfitted_conv = tf.keras.models.load_model('overfitted_conv_CIF10.model')
    except:
        print('overfitted conv model failed to load')
        fails += 1
        
    try:
        model_basic_nn = tf.keras.models.load_model('basic_nn_CIF10.model')
    except:
        print('basic nn model failed to load')
        fails += 1
        
    try: 
        model_overfitted_basic_nn = tf.keras.models.load_model('overfitted_basic_nn_CIF10.model')
    except:
        print('overfitted basic nn model failed to load')
        fails += 1
        
    if fails > 2:
        print('most models failed to load, retraining all models')
        import trainmodels as tm
        tm.train()
else:
    model_conv = tf.keras.models.load_model('conv_CIF10.model')
    model_overfitted_conv = tf.keras.models.load_model('overfitted_conv_CIF10.model')
    model_basic_nn = tf.keras.models.load_model('basic_nn_CIF10.model')
    model_overfitted_basic_nn = tf.keras.models.load_model('overfitted_basic_nn_CIF10.model')










#please note: only use the input line when you are running this from something that can take input,
#my IDE cannot give the input so it just hangs and crashes. use the second line in place of the image name.
#path = input("please type filename including the extension, no quotation marks, must be in same WorkingD as this .py file     ")
#path = 'test.jpg'
#path = str(path)

#path now has a default setup so if you have no path set it will go to a photos folder



#see reshape.py for details
imgs = re.reshapeAllImg()

#used for testing only
#plt.imshow(img, cmap=plt.cm.binary)

#indexable list of models and names, helpful for printing to screen
list_of_models = [('conv',model_conv),('overfitted_conv',model_overfitted_conv),('basic_nn',model_basic_nn),('overfitted_basic_nn',model_overfitted_basic_nn)]


#testing
#print(img.shape)

#if its a BW image we cant use our current models on it so this is just a type check ig. see concept.py for details on COLORED
#updated this, no more BW images.

ALL_MODELS = True

csv_array = np.empty(0, dtype=str)
temp_array = []
pred_array_for_test = []
for img,filename in imgs:
    #dont ask me why tensorflow wants a (1,32,32,3) array and not a (32,32,3) array but it does...
    img = tf.expand_dims(img, axis=0)
    #iterating over models for tagging
    if ALL_MODELS == True:
        for name, model in list_of_models:
            
            #call the actual prediction which returns a numpy array
            
            prediction = model_conv.predict(img,verbose=0)
            
            #print(prediction,'THIS IS PRED')
            #break
            #array of tags since we have a numpy array number not a name as output
            array_of_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
            pred_array_for_test.append(prediction)
            
            #turning our numpy val into the name of the tag
            list_for_multi_tags = prediction.tolist()
            tag1, tag2 = re.findVal(list_for_multi_tags[0])
            
            temp = array_of_names[tag1]
            if tag2:
                temp2 = array_of_names[tag2]
            
            if temp == filename:
                correct = 1
            else:
                correct = 0
            if tag2 == None or tag1 == tag2:
                new_tuple = (temp)
            else:
                new_tuple = (temp,temp2)
            
            
            #print to screen info including tag and model name
            csv_array = np.append(csv_array,new_tuple)
            temp_array.append(new_tuple)
            #print(f'this image is probably a {temp}, predicted by {name}')
            
            #testing only
            #plt.imshow(img[0], cmap=plt.cm.binary)
            #plt.show()




#jank ordered print for tags
for i in range(10):
    temp_a = []
    for x in range(10):
        temp_a.append(temp_array[10*i + x])
    print(array_of_names[i])
    print(temp_a)
    print('')
    
np.savetxt('tags.csv', csv_array,fmt="%10s", delimiter=',')



