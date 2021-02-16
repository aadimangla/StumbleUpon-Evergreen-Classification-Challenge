# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:52:02 2021

@author: Aditya Mangla
"""

# Importing Librries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from utils import boilerplate_model

#for creating feature column
from tensorflow.keras import layers
from tensorflow import feature_column
from os import getcwd

#Importing Training and Validation Dataset
ds = pd.read_csv('train1.csv')

#Importing Test Data
ds_test = pd.read_csv('test.csv')


# Pre processing test data
X_test = ds_test.iloc[:,3].values
X_test.reshape(1,-1)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X_test=labelencoder_X.fit_transform(X_test)

ds_test['alchemy_category'] = X_test
ds_test['alchemy_category_score'] = np.array(ds_test['alchemy_category_score'])


test_results = pd.read_csv('prediction.csv')
test_text_results = test_results.iloc[:,2].values
ds_test.pop('boilerplate')
ds_test.pop('url')
ds_test.pop('urlid')
ds_test.pop('news_front_page')
ds_test['boilerplate'] = np.array(test_text_results,dtype=float)
ds_test.info()




# Encoding categorical Variable
X = ds.iloc[:,3].values
X.reshape(1,-1)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X=labelencoder_X.fit_transform(X)

ds['alchemy_category'] = X

#Getting Boilerplate results using boilerplate Model
text_results = boilerplate_model()
text_results = np.array(text_results)
ds.pop('boilerplate')
ds.pop('url')
ds.pop('urlid')
ds.pop('news_front_page')
ds['boilerplate'] = text_results

ds.info()

train, val = train_test_split(ds, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')

#def df_to_dataset(dataframe, shuffle=True, batch_size=32):
#  dataframe = dataframe.copy()
#  labels = dataframe.pop('label')
#  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#  if shuffle:
#    ds = ds.shuffle(buffer_size=len(dataframe))
#  ds = ds.batch(batch_size)
#  return ds
train.info()
train_X = train
train_Y = np.array(train.pop('label'))
val_X= val
val_Y = np.array(val.pop('label'))
#batch_size = 64 # A small batch sized is used for demonstration purposes
#train_ds = df_to_dataset(train, batch_size=batch_size)
#val_ds = df_to_dataset(val, shuffle=False,batch_size=batch_size)

## Creating Feature Layer
#feature_columns = []
#
## Numeric Cols.
## Create a list of numeric columns. Use the following list of columns
## that have a numeric datatype:
#numeric_columns = ['alchemy_category','alchemy_category_score','avglinksize', 'commonlinkratio_1','commonlinkratio_2','commonlinkratio_3','commonlinkratio_4', 'compression_ratio','embed_ratio', 'framebased','frameTagRatio', 'hasDomainLink','html_ratio','image_ratio','is_news','lengthyLinkDomain', 'linkwordscore','non_markup_alphanum_characters','numberOfLinks','numwords_in_url','parametrizedLinkRatio','spelling_errors_ratio','boilerplate']
#
#for header in numeric_columns:
#    # Create a numeric feature column  out of the header.
#    numeric_feature_column = tf.feature_column.numeric_column(header)
#    
#    feature_columns.append(numeric_feature_column)
#
##feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    
    
# MODEL
model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=23, kernel_initializer='he_uniform', activation='tanh'),
        tf.keras.layers.Dense(128, activation='selu'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='selu'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.05, rho=0.9, momentum=0.2, epsilon=1e-07, centered=False,
    name='RMSprop'
), metrics=['accuracy','AUC',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

NUM_EPOCHS = 50
history = model.fit(x=train_X,y=train_Y, epochs=NUM_EPOCHS,validation_data=(val_X,val_Y))


results = model.predict(ds_test)
results = np.array(results)
results = np.round(results)
prediction = pd.DataFrame(results, columns=['predictions']).to_csv('prediction1.csv')


