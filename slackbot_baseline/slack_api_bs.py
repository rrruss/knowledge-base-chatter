# -*- coding: utf-8 -*-
"""Slack_Api_BS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12jHMlZPZceujMCH8j_Bu90UKjFO0c_to
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import urllib
import re

url = 'https://api.slack.com/faq'

response = requests.get(url)
page = response.text
soup = BeautifulSoup(page, "lxml")

Questions = []
for heading in soup.find_all(["h3"]):
    Questions.append(heading.text.strip())

answers = []
college_name_list = soup.find_all('h3')
for college in college_name_list:
  nextNode = college
  curr_answer = ""
  while True:
        nextNode = nextNode.find_next_sibling()
        if nextNode and nextNode.name != 'h1' and nextNode.name != 'h2'  and  nextNode.name != 'h3':
             curr_answer = curr_answer + nextNode.text
        else:
             answers.append(curr_answer)
             break
df_slack = pd.DataFrame(list(zip(Questions, answers)),
               columns =['Question', 'Answer'])

df_slack.to_csv('slack_api_qa.csv',index=False)
