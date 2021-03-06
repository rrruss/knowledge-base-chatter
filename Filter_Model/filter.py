# -*- coding: utf-8 -*-
"""Filter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_wX0ozp3l2slPIVgdRNdeZgDZe-PFnx
"""

import glob
import json
import re
from zipfile import ZipFile

import pandas as pd
import spacy
import yaml
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def create_json():
    with open("greetings.yml", 'r') as yaml_in, open("greetings.json", "w") as json_out:
         yaml_object = yaml.safe_load(yaml_in) # yaml_object will be a list or a dict
         json.dump(yaml_object, json_out)
    print("==============greetings.json created================")

create_json()

def create_greeting_df():
    f = open('greetings.json',)
    data = json.load(f)
    questions = []
    answers = []
    for i in data['conversations']:
       questions.append(i[0])
       answers.append(i[1])
    f.close()
    df_greeting = pd.DataFrame(list(zip(questions, answers)),
               columns =['Question', 'Answer'])
    print("===================greeting df created================")
    return df_greeting

#Small Talk Folder
with ZipFile('convo.zip', 'r') as zipObj:
   zipObj.extractall()

list_of_files = glob.glob("./convo8/*.txt")

def create_small_talk_df(list_of_files):
    questions = []
    answers = []
    lines = []
    for files in list_of_files:
        with open(files) as f:
             lines = f.readlines()
        count = 0
        for line in lines[7:-2]:
            count += 1
            if count%2 == 0: answers.append(line)
            else: questions.append(line)
    df_smalltalk = pd.DataFrame(list(zip(questions, answers)),
               columns =['Question', 'Answer'])
    print("=====================Small talk df created==================")
    return df_smalltalk

df_greeting = create_greeting_df()
df_smalltalk = create_small_talk_df(list_of_files)
frames = [df_greeting, df_smalltalk]
df_irrelevant = pd.concat(frames,ignore_index=True)
df_irrelevant['Query_type'] = 'irrelevant'

#Creating Relevant Query DF
df_relevant = pd.read_csv("slack_tips_qa_nodup.csv")

df_relevant.drop('Context',axis = 1,inplace=True)

df_relevant['Query_type'] = 'Relevant'

df_appleWatch = pd.read_csv("Apple_watch_tech_specs.csv")

df_appleWatch_rel = pd.DataFrame(columns=['Question', 'Answer', 'Query_type'])

df_appleWatch_rel['Question'] = df_appleWatch['watches_name'].str.replace("Technical Specifications", "") + df_appleWatch['watches_Features_name']

df_appleWatch_rel['Answer'] = df_appleWatch['watches_Features_values_name']

df_appleWatch_rel['Query_type'] = 'Relevant'

df_mac = pd.read_csv("Mac_tech_spec_all.csv")

df_mac.fillna({'product_features_name':"", 'product_features_Values_name':""}, inplace=True)

df_mac_rel = pd.DataFrame(columns=['Question', 'Answer', 'Query_type'])
df_mac_rel['Question'] = 'Apple ' + df_mac['product_name'].str.replace("Technical Specifications", "") + df_mac['product_features_name']
df_mac_rel['Answer'] = df_mac['product_features_Values_name']
df_mac_rel['Query_type'] = 'Relevant'

df_iphone = pd.read_csv("iphone_spec.csv")

df_iphone.fillna({'product_Feature_name':"", 'product_Feature_value_name':""}, inplace=True)

df_iphone_rel = pd.DataFrame(columns=['Question', 'Answer', 'Query_type'])
df_iphone_rel['Question'] = 'Apple ' + df_iphone['product_name'].str.replace("Technical Specifications", "") + df_iphone['product_Feature_name']
df_iphone_rel['Answer'] = df_iphone['product_Feature_value_name']
df_iphone_rel['Query_type'] = 'Relevant'

frames = [df_irrelevant, df_relevant,df_appleWatch_rel,df_mac_rel,df_iphone_rel]
df_combined = pd.concat(frames,ignore_index=True)
print("================Data is ready for pre-processing================")

#Pre-processing
nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser", "ner"])

def keep_token(t):
    # remove stop words, punct, numbers
    return (t.is_alpha and
            not (t.is_space or
                 t.is_stop or t.like_num))

def lemmatize_doc(doc):
    # Lemmatize
    return [ t.lemma_ for t in doc if keep_token(t)]

df_combined['clean_text'] = df_combined.Question.apply(lambda x: ' '.join(lemmatize_doc(nlp(str(x)))))
print("===================Data is ready for training==================")

#Split the Data in train/test
df_train, df_test = train_test_split(df_combined, test_size=0.33,shuffle=True, random_state=42)

X_train = df_train['clean_text'].values
y_train = df_train['Query_type'].values
X_test = df_test['clean_text'].values
y_test = df_test['Query_type'].values

#Training
classifier = LogisticRegression(max_iter = 500)
tfidf_vector = TfidfVectorizer()
pipe = Pipeline([('vectorizer', tfidf_vector),
                 ('classifier', classifier)])
pipe.fit(X_train,y_train)

# Model Training Accuracy
predicted_lr_train = pipe.predict(X_train)
print("Logistic Regression Training Accuracy:",metrics.accuracy_score(y_train, predicted_lr_train))

# Model Test Accuracy
predicted_lr_test = pipe.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted_lr_test))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted_lr_test,average='weighted'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted_lr_test,average='weighted'))


#chat = "Hi. How are you ?. How is the weather today. Is slack Hung ? How to send email from slack. What are the features of Apple iphone X ? I am a fool."
chat = input('Enter the Query: ')
def extract_relevant_conversation(chat,pipe):
   result = []
   chat_sentences = re.findall(r'[^.!?\n]+[.!?]', chat)
   for sentence in chat_sentences:
       clean_text = ' '.join(lemmatize_doc(nlp(str(sentence))))
       if clean_text:
          if (pipe.predict([clean_text])) == 'Relevant' :
            result.append(sentence)
   return result

#chat = "Hi. How are you ?. How is the weather today. Is slack Hung ? How to send email from slack. What are the features of Apple iphone X ? I am a fool."
answer = extract_relevant_conversation(chat,pipe)

print(answer)
