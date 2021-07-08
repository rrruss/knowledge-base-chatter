# Knowledge Retrieval Framework using T5 and Bert

This github has all the pieces for creating a QA bot from support websites of a company. It is using slack and apple as an example. 

## Pipeline
  ![Pipeline](./Pipleline.png)
  
## Different Pieces

### Scraper (KB generation)
Code for scraping is in scraping folder. scraping_bs.py was used for scraping slack tips in csv with columns: question, answer,context, answer_start.
This csv was then fed to T5 for finetuning after pre-processing (Explained in T5 section). 
Apple support site scraping was done through parse hub and scrapy crawler. Data was directly loaded in s3. This piece is not included in github.

### Filter Model
This model first splits the input queries in sentences and then each sentence is classified as relevant/irrelevant. Only relevant portion of the query is passed
along.
Code is under Filter_Model/filter.py. This file can just be run as standalone python filter.py and it asks for an input query and will respond back with only 
relevant portion

### T5
This portion has been taken from this github :- https://github.com/patil-suraj/question_generation
Scraped slack data was pre-processed and used for finetuning. And then QA task was used to generate question answer pairs.

### Reader/Retriever Model
