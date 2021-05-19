"""Slack API Information chat bot"""

import os
import re
import time
import sys
import inspect
import random

import spacy
import swifter
import nltk
import ssl
import spacy_universal_sentence_encoder
import pandas as pd

from colorama import init
from termcolor import cprint
from pyfiglet import figlet_format
from spacy import displacy
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer

# Had to add below try statement (Otherwise getting error while trying to download from nltk)
try:

    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

########################################################################################
# slack-qa csvs
SLACK_QA_DATA = ('%s' % (os.path.join(os.path.dirname(__file__), './slack_api_qa.csv')))

# For sentence conversions
CONTRACTIONS_DICT = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

########################################################################################
# For logging
import logging

# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)
log.propagate = False
########################################################################################


class SlackQAChatbot(object):
    """Slack API Information chat bot """
    def __init__(self):
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        self.set_dafa_frame_from_csv(
            [SLACK_QA_DATA]
        )
        self.load_model()
        self.create_clean_title_column()
        log.debug(f"{self.df[['title', 'clean_title', 'Answer']]}")
        self.create_lemmatized_question_answer_column()
        self.nlp_for_user_question = spacy.load('en_core_web_lg')
        # Sample question (to create similarity column) - > next time onwards it gets fast
        self.get_answer_for_most_similar_title('can pets have corona')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def set_dafa_frame_from_csv(self, csvs=None):
        """Sets data frame based on the csvs passed

        :param list csvs: list of path of csv files

        :returns pd.dataFrame: sets data frame based on the csv passed

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        if (not isinstance(csvs, list)):
            list(csvs)

        # These are the relevant columns needed for chat bot (Rest I will remove)
        # And these are the common columns available in all csvs
        columns_to_keep = ['title', 'Question', 'Answer']
        df_list = []

        for csv in csvs:
            df_temp = pd.read_csv(csv)
            log.debug(f'csv: {csv}, shape {df_temp.shape}')
            # If title column not in csv then create one as duplicate of question column
            if ('title' not in df_temp.columns):
                df_temp['title'] = df_temp['Question']

            # Multilingual csv has language column, keep only rows that has language as english
            if ('language' in df_temp.columns):
                df_temp = df_temp[df_temp['language'].isin(['english', 'English', 'ENGLISH'])]

            # Now remove all columns except columns_to_keep
            df_temp = df_temp[columns_to_keep]
            df_list.append(df_temp)

        if (len(df_list) > 1):
            self.df = pd.concat(df_list)
        else:
            self.df = df_list[0]

        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)
        log.debug(f'shape:{self.df.shape}')
        log.debug(f'df columns:{self.df.columns.to_list()}')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def load_model(self):
        """Loads the model"""
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # This library lets you use Universal Sentence Encoder embeddings of Docs, Spans and Tokens
        # directly from TensorFlow Hub
        self.nlp_for_sent_similarity = spacy_universal_sentence_encoder.load_model('en_use_lg')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def clean_sentence(self, sentence, lemmatize=False):
        """Clean the sentence

        :param string sentence: sentence to be cleaned
        :param boolean lemmatize: lemmatize sentence if True else Don't

        :returns string: clean sentence

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # Cleaning sentence
        new_string = str(sentence).lower()
        new_string = re.sub(r'\([^)]*\)', '', new_string)
        new_string = re.sub('"', '', new_string)
        new_string = ' '.join([CONTRACTIONS_DICT[t] if t in CONTRACTIONS_DICT else t for t in
                              new_string.split(" ")])
        new_string = re.sub(r"'s\b", "", new_string)
        new_string = re.sub("[^a-zA-Z]", " ", new_string)

        words = word_tokenize(new_string)

        # For reducing words to their root form
        lemma = WordNetLemmatizer()

        # if lemmatize True then lemmatize sentence and remove stopwords
        if (lemmatize):
            stop_words = set(stopwords.words('english'))
            words = [lemma.lemmatize(word, 'v') for word in words if (word) not in stop_words]

        return (" ".join(words)).strip()

    def sentence_similarity(self, sentence_1, sentence_2):
        """Get similarity between two sentences

        :param string sentence_1: sentence 1
        :param string sentence_1: sentence 2

        :returns float: similarity index

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # USed to get similarity between user question and slack database questions
        sentence_1 = self.nlp_for_sent_similarity(sentence_1)
        sentence_2 = self.nlp_for_sent_similarity(sentence_2)
        return sentence_1.similarity(sentence_2)

    def create_clean_title_column(self):
        """Create a new column in dataframe 'clean_title' with cleaned sentences from column 'title'

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        self.df['clean_title'] = self.df['title'].swifter.apply(self.clean_sentence)
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def create_lemmatized_question_answer_column(self):
        """Create new columns in dataframe 'clean_question' and 'clean_answer' with cleaned
        sentences with lemmatization from column 'question' and 'answer' respectively

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        self.df['clean_question'] = self.df['Question'].swifter.apply(
            self.clean_sentence,
            args=(True,)
        )
        self.df['clean_answer'] = self.df['Answer'].swifter.apply(
            self.clean_sentence,
            args=(True,)
        )
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def create_sentence_similarity_column(self, user_question):
        """Create a new column in dataframe 'sim' that will correspond to similarity between user
        question and dataframe questions (using column 'title') for comparison. This similarity will
        be used to later help in algorithm to select answer best matched for user question

        :param string user_question: user question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        self.df['sim'] = ''
        user_question = self.clean_sentence(user_question)
        self.df['sim'] = self.df['clean_title'].swifter.apply(
            self.sentence_similarity,
            args=(user_question,)
        )
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def get_nouns_verbs_adj_from_user_question(self, user_question):
        """Returns all nouns, verbs and adjectives in user question

        :param string user_question: user question

        :returns list: nouns, verbs and adjectives in user question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        user_question = self.clean_sentence(user_question)
        lemma = WordNetLemmatizer()
        noun_verb_list = [
            ent.text for ent in self.nlp_for_user_question(user_question) if (ent.pos_ in ['NOUN', 'VERB', 'ADJ'])
        ]
        return [lemma.lemmatize(word, 'v') for word in noun_verb_list]

    def check_user_question_nouns_in_df_answer_and_question(self, user_question_nouns, index):
        """Checks if user question's nouns, verbs and adjective in 'clean_answer' and
        'clean_question' column at index 'index' in df

        :param list user_question_nouns: nouns, verbs and adjectives in user question
        :param int index: index of df

        :returns boolean: True if any of the nouns, verbs and adjectives found in clean_answer' or
        'clean_question' column at index 'index' in df else False

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        log.debug(f"clean_answer: {self.df._get_value(int(index), 'clean_answer')}, index: {index}")
        log.debug(
            f"clean_question: {self.df._get_value(int(index), 'clean_question')}, index: {index}"
        )
        if (any([noun in self.df._get_value(int(index) ,'clean_answer') for noun in user_question_nouns])
                or any([noun in self.df._get_value(int(index), 'clean_question') for noun in user_question_nouns])):
            return True

        return False

    def get_answer_for_most_similar_title(self, user_question):
        """Checks the dataframe and looks for most similar question in df to user question

        :param string user_question: user question

        :returns string: best suited answer for user's question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        log.debug('Creating question similarity column')
        self.create_sentence_similarity_column(user_question)

        df_copied = self.df.copy()
        # sort index values based on the decreasing order of similarity (using 'sim' column we
        # created by checking similarity against user question)
        df_copied = df_copied.sort_values(by='sim', ascending=False)

        # Get sorted index values of df in a list
        sorted_index_values_based_on_sim = df_copied.index.values.tolist()

        # For debugging print below info
        log.debug(f"{df_copied[['title', 'clean_title', 'Answer', 'sim', 'clean_answer', 'clean_question']]}")
        log.debug(f'sorted_index_values_based_on_sim: {sorted_index_values_based_on_sim}')

        # Get all nouns, verbs and adjectives in user question as a list
        user_question_nouns_verbs_adj = self.get_nouns_verbs_adj_from_user_question(user_question)
        log.debug(
            f'user_question_nouns: {self.get_nouns_verbs_adj_from_user_question(user_question)}'
        )

        # If no nouns, verbs and adjectives in user question, return Sorry...Invalid question!
        if (not user_question_nouns_verbs_adj):
            return 'Sorry...Invalid question!'

        # Now iterate df based on sorted index values and check for which index we find user
        # question's nouns, verbs and adjective in 'clean_answer' or 'clean_question' column
        for index in sorted_index_values_based_on_sim:
            if (
                    self.check_user_question_nouns_in_df_answer_and_question(
                        user_question_nouns_verbs_adj, index
                    )
            ):
                return self.df._get_value(int(index) ,'Answer')

        # Return 'Sorry...No suitable answer available in database!' if no suitable answer found
        return 'Sorry...No suitable answer available in database!'

    # Todo: If time permits
    def print_article_summary_with_entities(self):
        pass

    def print_chatbot_ready_text(self, text):
        """Prints fancy 'Slack CHATBOT' when program starts"""
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        init(strip=not sys.stdout.isatty())
        cprint(
            figlet_format(
                text
            ),
            color='yellow',
            attrs=['bold', 'blink']
        )
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def get_letter_at_random_interval(self, answer):
        """Retrieves letter from answer at random interval"""
        answer = re.sub(r'\n+|\r+', '\n', answer).strip()
        answer = '\n'.join([ll.rstrip() for ll in answer.splitlines() if ll.strip()])

        for letter in answer:
            # Adding below random time sleep to give an illusion that it is thinking
            rand_num = random.randrange(1, 100, 1)

            if (rand_num > 98):
                time.sleep(random.randrange(1, 1000, 1) / 1000.0)
            elif (rand_num > 96):
                time.sleep(random.randrange(1, 500, 1) / 1000.0)
            else:
                time.sleep(random.randrange(1, 3, 1) / 1000.0)

            yield letter


###########################################################################################

# Creating Slack Chatbot GUI with tkinter
import random
from tkinter import *

def main():
    """main function that starts the slack api information chatbot"""
    log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
    ml = SlackQAChatbot()
    ml.print_chatbot_ready_text('SLACK API CHATBOT BY CISCO Chatter Group')
    welcome_text = 'Hi, I am slack api bot. You can ask me any questions related to slack api issues!!!'

    def send():
        """Sends text to the text box widget for user question and answer for the user question"""
        user_question = EntryBox.get("1.0", 'end-1c').strip()
        EntryBox.delete("0.0", END)
        ChatLog.config(state=NORMAL)
        if (user_question != ''):
            ChatLog.insert(END, user_question + '\n\n', 'you_text')
            ChatLog.update()

            ChatLog.insert(END, "Bot: ", 'bot')
            ChatLog.update()

            # Get answer for the user question
            answer = ml.get_answer_for_most_similar_title(user_question)

#            for letter in ml.get_letter_at_random_interval(answer):
            for letter in answer:
                ChatLog.insert(END, letter, 'bot_text')
                ChatLog.update()
                ChatLog.yview(END)

            ChatLog.insert(END, '\n\n', 'bot_text')
            ChatLog.insert(END, "You: ", 'you')
            ChatLog.update()
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

    base = Tk()
    base.title("Slack API Information Bot")
    base.geometry("1100x700")
    base.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    ChatLog = Text(base, bd=0, bg="black", height="8", width="50", font="Arial", )
    ChatLog.config(state=DISABLED)
    ChatLog.tag_config('you', foreground="#ffa500", font=("Ariel", 14, "bold"))
    ChatLog.tag_config('bot', foreground="#7cec12", font=("Ariel", 14, "bold"))
    ChatLog.tag_config('you_text', foreground="#ffa500", font=("Verdana", 13))
    ChatLog.tag_config('bot_text', foreground="#7cec12", font=("Verdana", 13))

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview)
    ChatLog['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(
        base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0,
        highlightbackground="#32de97",
        highlightcolor="#008000", fg='#000000', command=send
    )

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial",
                    selectborderwidth=2)

    # Place all components on the screen
    scrollbar.place(x=1076, y=6, height=586)
    ChatLog.place(x=6, y=6, height=586, width=1070)
    SendButton.place(x=6, y=601, height=90)
    EntryBox.place(x=128, y=601, height=90, width=965)

    EntryBox.focus_set()

    # Insert welcome text
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "Bot: ", 'bot')
    ChatLog.insert(END, welcome_text + '\n\n', 'bot_text')
    ChatLog.insert(END, "You: ", 'you')
    ChatLog.config(state=DISABLED)
    ChatLog.update()

    base.mainloop()

if __name__ == '__main__':
    try:

        main()

    except KeyboardInterrupt:
        log.critical('Keyboard Interrupted!!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
