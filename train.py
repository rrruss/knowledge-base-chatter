import re
import string

import pandas as pd

from dataloader.dataloaders import dataloader
from models.retrievalmodel import LongQAModel
from trainer.trainers import Trainer

if __name__ == '__main__':

    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    # get Slack data
    # df = pd.read_json('https://www.dropbox.com/s/r7iwb6qpk73jhrk/qas.json?dl=1')
    df1 = pd.read_csv('https://www.dropbox.com/s/by3dcp6y07g3g2q/slack_samples.csv?dl=1')
    df2 = pd.read_csv('https://www.dropbox.com/s/30jgxxwkyy1ywir/slack_tips_qa_nodup_answer_start.csv?dl=1')
    df2['question'] = df2['Question']
    df2['answer'] = df2['Answer']
    df2['context'] = df2['Context']
    df = df1.append(df2[['question', 'answer', 'context']]).sample(frac=1.)

    # clean questions, answers, and contexts
    qa_dicts = df.to_dict(orient='records')
    for d in qa_dicts:
        for key in d:
            if key != 'mlm':
                d[key] = ''.join(
                    c for c in d[key] if c in string.ascii_letters + string.digits + string.punctuation + ' ')
                d[key] = re.sub(r' {2,}', ' ', d[key])

    # instantiate model
    model = LongQAModel(contexts=[d['context'][:2000] for d in qa_dicts],)

    # get data loader
    train_dataloader = dataloader(qa_dicts,
                                  fast_tokenizer=model.r_tokenizer,
                                  split='train',
                                  batch_size=32,
                                  train_size=0.98)
    valid_dataloader = dataloader(qa_dicts,
                                  fast_tokenizer=model.r_tokenizer,
                                  split='valid',
                                  batch_size=32,
                                  train_size=0.98)

    # train model
    trainer = Trainer(model,
                      submodule_to_train='r_model',
                      tokenizer=model.r_tokenizer,
                      dataloader=train_dataloader,
                      validation_dataloader=valid_dataloader,
                      lr=1e-5,
                      batch_size=2,
                      epochs=251)
    trainer.train()
