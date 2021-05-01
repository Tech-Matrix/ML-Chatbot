from nltk import metrics, stem
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
import tflearn
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
import json
import random
import pickle
import os
import matplotlib.pyplot as plt

with open("WolbotDataset.json") as file:
    data = json.load(file)

words=[]
labels=[]
train_x=[]
train_y=[]
#storing the intents and their respective patterns
for intents in data['intents']:
    for patterns in intents['patterns']:
        patterns = patterns.lower()
        #spliting the each pattern into space seperated words by removing punctuations(Tokenization)
        wrds=nltk.word_tokenize(patterns)
        words.extend(wrds)
        train_x.append(wrds)
        train_y.append(intents['tag'])
    if intents['tag'] not in labels:
        labels.append(intents["tag"])   

stemmer = SnowballStemmer("english")
words=[stemmer.stem(w.lower()) for w in words if w!='?']
words=sorted(list(set(words)))
labels=sorted(labels)

training=[]
output=[]
output_rep=[0 for x in range(len(labels))]
for x,doc in enumerate(train_x):
    wrds=[stemmer.stem(w) for w in doc]
    b=[1 if j in wrds else 0 for j in words]
    output_row=output_rep[:]
    output_row[labels.index(train_y[x])]=1
    training.append(b)
    output.append(output_row)
####converting the data into arrays
training=np.array(training)
output=np.array(output)
#### storing the data
with open("data.pickle","wb") as f:
    pickle.dump((words,labels,training,output),f)
####
net = tflearn.input_data(shape = [None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch = 500, batch_size = 8, show_metric = True)
model.save("chatbot1_model")


def sentencestem(sentence,words):
    wrds=nltk.word_tokenize(sentence)
    b=[0 for w in words]
    stemmer = SnowballStemmer("english")
    s_words=[stemmer.stem(word.lower()) for word in wrds]
    for s in s_words:
        for i,w in enumerate(words):
            if w==s:
                b[i]=1
    return np.array(b)  

def predict_label(text,model):
    ans=sentencestem(text,words)
    res=model.predict([ans])
    result_index=np.argmax(res)
    tag=labels[result_index]
    for tg in data["intents"]:
            if tg["tag"]==tag:
                responses=tg['responses']
                res =random.choice(responses)
    return res            

def chatbot_response(text):
    ints =predict_label(text, model)
    return ints 

#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You:" + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    res = chatbot_response(msg)
    ChatLog.insert(END, "ChatBot:" + res + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)
base = Tk()
base.title("ChatBot")
base.geometry("400x500")
base.resizable(width=False, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)


#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#de3252", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=380,y=3, height=320)
ChatLog.place(x=6,y=6, height=320, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()

