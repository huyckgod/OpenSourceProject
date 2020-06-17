#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import cv2  
import matplotlib.pyplot as plt  

# define function to load train, test, and validation datasets
def load_dataset(path):
    '''
    Load dataset
    '''
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']),24)
    #print(data['filenames'])
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

# load list of dog names
dog_names = [item[11:-1] for item in sorted(glob("data/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


# In[5]:


from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
#restnet.summary()


# In[6]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()

model.add(restnet)

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(24, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
#model.summary()


# In[7]:


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[8]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
#train_tensors = paths_to_tensor(train_files).astype('float32')/255
#valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
#test_tensors = paths_to_tensor(valid_files).astype('float32')/255


# In[9]:


#from keras.callbacks import ModelCheckpoint  

#checkpointer = ModelCheckpoint(filepath='saved_models/weights.hdf5', 
#                               verbose=1, save_best_only=True)


# In[10]:


#model.fit(train_tensors, train_targets, 
#          validation_data=(valid_tensors, valid_targets),
#          epochs=5, batch_size=20, verbose=1, callbacks=[checkpointer])


# In[11]:


model.load_weights('saved_models/weights.hdf5')


# In[41]:


from keras.applications.resnet50 import preprocess_input, decode_predictions

ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def dog_breed(img_path):
    predicted_vector = model.predict(path_to_tensor(img_path)) #shape error occurs here
    #return dog breed that is predicted by the model
    #print(predicted_vector)
    fir = dog_names[np.argmax(predicted_vector)]
    fir_per = np.max(predicted_vector) * 100
    fir_index = np.argmax(predicted_vector)
    predicted_vector[0][fir_index] = 0
    
    sec = dog_names[np.argmax(predicted_vector)]
    sec_per = np.max(predicted_vector) * 100
    
    return fir,fir_per,  sec,sec_per


# In[28]:


### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def dog_breed_a(image_pth):
    if dog_detector(image_pth):
        img = cv2.imread(image_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img);
        breed_1,per1,breed_2,per2 = dog_breed(image_pth)
        persum= per1 + per2
        print('this dog is,', breed_1,per1/persum *100, "%",'or ' , breed_2, per2/persum*100,"%")
    else:
        img = cv2.imread(image_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img);
        print('This is not a dog.')
        breed_1,per1,breed_2,per2 = dog_breed(image_pth)
        persum= per1 + per2
        print('If it is dog,', breed_1,per1/persum *100, "%",'or ' , breed_2, per2/persum*100,"%")


# In[42]:


dog_breed_a('images/17.jpg')


# In[17]:


dog_names


# In[ ]:




