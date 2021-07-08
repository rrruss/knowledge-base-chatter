import argparse
import json
import random
import string
from datetime import date
from typing import Optional

import boto3
import demoji
import pandas as pd


demoji.download_codes()


S3_BUCKET = "kbchatter"
S3_OUTPUT_PATH = "scraped_data/slack"

slackcsv = pd.read_csv(
    "https://kbchatter.s3.amazonaws.com/scraped_data/slack/slack_tips_qa_nodup_answer_start.csv"
)
slackcsv.drop_duplicates(inplace=True)


def clean_text(s: str) -> str:
    """
    1. check for space after : and add one if it does not exist
    2. replace newline \n with space
    3. replace \xa0 with space
    4. replace emoji with space
    """
    newstr = demoji.replace(s, " ")
    newstr = newstr.replace("\xa0", " ")
    newstr = newstr.replace("\n", " ")
    newstr_list = []
    for idx, char in enumerate(newstr):
        if char == ":" and idx == len(newstr) - 1:
            return "".join(newstr_list)
        newstr_list.append(char)
        if char == ":" and newstr[idx + 1] != " ":
            newstr_list.append(" ")

    return newstr


def squad_format(context: str, df: pd.DataFrame) -> dict:
    context_dict = {"context": clean_text(context), "qas": []}
    for idx, row in df.iterrows():
        question_dict = {
            "question": clean_text(row["Question"]),
            "id": "".join(random.choices(string.ascii_letters + string.digits, k=16)),
            "answers": [
                {"answer_start": row["answer_start"], "text": clean_text(row["Answer"])}
            ],
        }
        context_dict["qas"].append(question_dict)

    return context_dict


def main(upload: bool) -> Optional[dict]:

    total_num_context = len(slackcsv["Context"].unique())
    num_train = 2 * total_num_context // 3
    num_dev = total_num_context - num_train

    unique_contexts = slackcsv["Context"].unique()

    squad_formatted_train = {"data": [{"title": "train test", "paragraphs": []}]}
    for c in unique_contexts[:num_train]:
        df = slackcsv.query("Context == @c")
        squad_formatted = squad_format(c, df)
        squad_formatted_train["data"][0]["paragraphs"].append(squad_formatted)

    squad_formatted_dev = {"data": [{"title": "dev test", "paragraphs": []}]}
    for c in unique_contexts[-num_dev:]:
        df = slackcsv.query("Context == @c")
        squad_formatted = squad_format(c, df)
        squad_formatted_dev["data"][0]["paragraphs"].append(squad_formatted)

    if upload:
        today = date.today().strftime("%Y%m%d")
        s3 = boto3.client("s3")
        s3.put_object(
            Body=json.dumps(squad_formatted_train),
            Bucket=S3_BUCKET,
            Key=f"{S3_OUTPUT_PATH}/squad_formatted_train_{today}.json",
        )
        s3.put_object(
            Body=json.dumps(squad_formatted_dev),
            Bucket=S3_BUCKET,
            Key=f"{S3_OUTPUT_PATH}/squad_formatted_dev_{today}.json",
        )
    else:
        print("Training set:")
        print(json.dumps(squad_formatted_train))
        print("Dev set:")
        print(json.dumps(squad_formatted_dev))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format data in SQUAD format for fine tuning"
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help="Format data into SQUAD json and upload to S3.",
    )
    parser.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="Format data into SQUAD json without uploading to S3",
    )
    parser.set_defaults(upload=False)
    args = parser.parse_args()

    main(upload=args.upload)
