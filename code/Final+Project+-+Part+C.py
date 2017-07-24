
# coding: utf-8

# In[463]:

from keras.utils.data_utils import get_file
path = get_file('trump3.txt', origin="https://raw.githubusercontent.com/galst/FinalProject/master/data/trump-txt.txt")
text = open(path).read()
print('corpus length:', len(text))


# In[464]:

text[0:20000]


# In[465]:

vocabulary_size = 600
unknown_token = "UNKNOWNTOKEN"


# In[466]:

sentence_start_token = "SENTENCESTART"
sentence_end_token = "SENTENCEEND"


# In[467]:

separator= "SEPARATOR"


# In[468]:

text1 = text.replace('\n', ' ')
text1 = text1.replace('--',' '+ separator + ' ')
text1 = text1.replace('.',' '+sentence_end_token +' '+ sentence_start_token+' ' )
text1[0:2000]


# In[469]:

from keras.preprocessing.text import text_to_word_sequence
text2 = text_to_word_sequence(text1, lower=False, split=" ") #using only 10000 first words


# In[470]:

text2[0:100]


# In[471]:

from keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=600,char_level=False)
token.fit_on_texts(text2)


# In[472]:

text_mtx = token.texts_to_matrix(text2, mode='binary')


# In[473]:

text_mtx.shape


# In[474]:

input_ = text_mtx[:-1]
output_ = text_mtx[1:]


# In[475]:

input_.shape, output_.shape


# In[476]:

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


# In[477]:

model = Sequential()
model.add(Embedding(input_dim=input_.shape[1],output_dim= 42, input_length=input_.shape[1]))
# the model will take as input an integer matrix of size (batch, vocabulary_size).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, vocabulary_size, 42), where None is the batch dimension.


# In[478]:

model.add(Flatten())
model.add(Dense(output_.shape[1], activation='sigmoid'))


# In[479]:

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
model.fit(input_, y=output_, batch_size=300, epochs=10, verbose=1, validation_split=0.2)


# In[480]:

import numpy as np
def get_next(text,token,model,fullmtx,fullText):
    tmp = text_to_word_sequence(text, lower=False, split=" ")
    tmp = token.texts_to_matrix(tmp, mode='binary')
    p = model.predict(tmp)
    best10 = p.argsort() [0][-10:]
    bestMatch = np.random.choice(best10,1)[0]
    next_idx = np.min(np.where(fullmtx[:,bestMatch]>0))
    return fullText[next_idx]


# In[481]:

import random
text3 = ''
for x in range(0, 30):
    currWord = random.choice(text2)
    while currWord == unknown_token or currWord == sentence_start_token or currWord == sentence_end_token or currWord == separator:
        currWord = random.choice(text2)
    text3 = text3 + currWord
    for y in range(0, 20):
        nextWord = get_next(currWord,token,model,text_mtx,text2)
        if nextWord == sentence_start_token or nextWord == sentence_end_token:
            text3 = text3 + ' ' + sentence_end_token + ' ' + sentence_start_token + ' '
        else:
            text3 = text3 + ' ' + nextWord
        currWord = nextWord
        if y == 19:
            text3 = text3 + '.\n'
text3


# In[482]:

text3 = text3.replace(' '+ separator + ' ','--')
text3 = text3.replace(' ' +sentence_end_token +' '+ sentence_start_token + ' ', '.')
text3


# In[483]:

new_path = 'C:/src/Academy/final/FinalProject/data/trump-output-txt.txt'
new_file = open(new_path,'w')
new_file.write(text3)
new_file.close()


# In[ ]:



