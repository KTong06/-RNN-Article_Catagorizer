# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:16:39 2022

@author: KTong
"""

import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import Visuals as vs
import NN_modules as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.utils import plot_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#%% STATICS
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(),'logs',log_dir)
URL_PATH='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_PATH=os.path.join(os.getcwd(),'saved_models','tokenizer_textseg.json')
OHE_PKL_PATH=os.path.join(os.getcwd(),'saved_models','ohe_textseg.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','model.h5')
vocab_size=18000
oov_token='OOV'
max_len=205


#%% DATA LOADING
df=pd.read_csv(URL_PATH)

#%% DATA INSPECTION / VISUALIZATION
df.head()
df.info()

# Check for duplicates
df.duplicated().sum()
df[df.duplicated()]

# 99 rows returned as duplicate however upon further inspection the claimed 
# duplicates contain different articles, hence will not be discarded.

# Visualize distribution of category
target=['category']
visuals=vs.Visualisation()
visuals.cat_plot(df,target)

#%% DATA CLEANING
text=df['text'].values # Features : X
category=df['category'].values # category : Y

# Load English stop words
stop_words=set(stopwords.words('english'))

# Instantialize lemmatizer
lemmatizer=WordNetLemmatizer()

# Remove numbers, symbols, stop words and filter morphed words in text
for index,texts in enumerate(text):
    text[index]=re.sub('[^a-zA-Z]',' ',texts).lower().split()
    text[index]=[w for w in text[index] if not w in stop_words]
    text[index]=[lemmatizer.lemmatize(w) for w in text[index]]

#%% PREPROCESSING
# TOKENISATION
tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)

# To check number of words registered
word_index=tokenizer.word_index
print(word_index)

# 'word_index' contains 24742 after filtering stop words and 
# applied lemmatization, taking 75% of 'word_index' as 'vocab_size'

# Tokenization
train_sequences=tokenizer.texts_to_sequences(text)

# Export tokenizer in json format
token_json=tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)


# PADDING AND TRUNCATING
# Find average number of words in texts
length_of_text=[len(i) for i in train_sequences]
print(np.median(length_of_text)) 
print(np.mean(length_of_text)) 
visuals.single_cont_plot(length_of_text)

# Distribution of text length is skewed.
# median text length:191, average text length:218, maxlen=(190+220)/2

# Apply padding to text
padded_text=pad_sequences(train_sequences,maxlen=max_len,truncating='post',
                          padding='post')


# ONE HOT ENCODING FOR TARGET COLUMN
ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))

# Export ohe
with open(OHE_PKL_PATH,'wb') as file:
    pickle.dump(ohe,file)

# ohe.inverse_transform(np.expand_dims(category[5],axis=0))
# 4-tech,0-bussiness,3-sport,1-entertainement,2-politics

# TRAIN TEST SPLIT
x_train,x_test,y_train,y_test=train_test_split(padded_text,category,test_size=0.3,random_state=123)

#%% MODEL DEVELOPMENT
nn_mod=nn.NeuralNetworkModel()
model=nn_mod.RNN_model(x_train,y_train,mask=0,vocab=vocab_size,
                       embed_dim=128,l2_node=32,droprate=0.2)

plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss=('categorical_crossentropy'),
              metrics=['acc'])

tb=TensorBoard(log_dir=LOG_PATH)
es=EarlyStopping(monitor='val_loss',patience=10)

hist=model.fit(x_train,y_train,batch_size=128,epochs=50,
               validation_data=(x_test,y_test),verbose=2,callbacks=[tb,es])

#%% MODEL EVALUATION
nn_mod.eval_plot(hist)
nn_mod.model_eval(model,x_test,y_test,label=[0,1,2,3,4])

#%% MODEL EXPORT
model.save(MODEL_SAVE_PATH)


