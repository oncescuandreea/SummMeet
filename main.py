import argparse
from pathlib import Path

import openai
import tqdm
from huggingface_hub import notebook_login
from pyannote.audio import Pipeline
from transformers import pipeline

from rouge import Rouge
from utils import *
import streamlit as st

openai.api_key = "<INSERT YOUR OWN KEY HERE>"


def dizarization_fct(pipeline_diar, file_path, file_name, desired_audio_type):
    """
    Extract start and end times for each speaker and the corresponding speakers
    for each time interval.
    """
    diarization = pipeline_diar(
        str(file_path.parent / f"{file_name}.{desired_audio_type}")
    )
    start_end_times = []
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        start_end_times.append((turn.start, turn.end))
        speakers.append(speaker)
    return start_end_times, speakers


def using_BART(no_tokens: int, max_tokens_model: int, str_trans_sp: dict, sum_model: str):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        summary = summarizer(
            f'""{str_trans_sp}""', max_length=50, min_length=10, do_sample=False
        )
        # print(f"When using {sum_model} summary is:\n {summary}")
    else:
        # print(f"There are more tokens than supported by the model.")
        st.write(f"There is more context in the transcript than this model can take in. Will slide the model over the "
                 f"allowed context length.")
        summaries = []
        for sentence in str_trans_sp:
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                summary = summarizer(
                    f'""{transcription_chunk}""',
                    max_length=30,
                    min_length=10,
                    do_sample=False,
                )
                summaries.append(summary[0]["summary_text"])
                # print(f"Current summary is {summary[0]}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())

        if transcription_chunk != "":
            summary = summarizer(
                f'""{transcription_chunk}""',
                max_length=30,
                min_length=10,
                do_sample=False,
            )
            summaries.append(summary[0]["summary_text"])
        # print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        # print(f"no tokens in concat summary is {len(concat_summaries.split())}\n")
        summary = summarizer(
            f'""{transcription_chunk}""',
            max_length=150,
            min_length=50,
            do_sample=False,
        )
        # print(f"When using {sum_model} final summary is:\n {summary}")
        st.write("##### Final summary is:")
        st.write(f"#### {summary[0]['summary_text']}")
    with open(f"data/bart_summaries/experiment.txt", "w") as f:
        f.write(summary[0]["summary_text"])
    return summary[0]["summary_text"]


def using_GPT3(no_tokens, max_tokens_model, str_trans_sp, sum_model):
    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{str_trans_sp}\nTl;dr",
            temperature=0.3,
            max_tokens=120,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(
            f"When using {sum_model} summary is:\n\
                {response.choices[0].text}"
        )
    else:
        print(f"There are more tokens than supported by the model.")
        summaries = []
        for sentence in str_trans_sp:
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=f"{transcription_chunk}\nTl;dr",
                    temperature=0.3,
                    max_tokens=120,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                summaries.append(response.choices[0].text)
                print(f"Current summary is {response.choices[0].text}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())

        if transcription_chunk != "":
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"{transcription_chunk}\nTl;dr",
                temperature=0.3,
                max_tokens=120,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            summaries.append(response.choices[0].text)
            print(f"Current summary is {response.choices[0].text}")
        print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        print(f"no tokens in concat summary is {len(concat_summaries.split())}\n")
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{transcription_chunk}\nTl;dr",
            temperature=0.6,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(
            f"When using {sum_model} summary is:\n\
                {response.choices[0].text}"
        )
        response_sum = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"{transcription_chunk}\nSummarize",
            temperature=0.6,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(
            f"When using {sum_model} summary is:\n\
                {response_sum.choices[0].text}"
        )


def using_Longformer(no_tokens, max_tokens_model, str_trans_sp, sum_model):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    tokens_used_so_far = 0
    transcription_chunk = ""
    if no_tokens < max_tokens_model:
        summary = summarizer(
            f'""{str_trans_sp}""', max_length=50, min_length=10, do_sample=False
        )
        # print(f"When using {sum_model} summary is:\n {summary}")
    else:
        print(f"There are more tokens than supported by the model.")
        summaries = []
        for sentence in str_trans_sp:
            if tokens_used_so_far + len(sentence.split()) <= max_tokens_model:
                transcription_chunk += sentence
                tokens_used_so_far += len(sentence.split())
                to_generate_summary = True
                # print(f"Got here with {tokens_used_so_far}")
            else:
                summary = summarizer(
                    f'""{transcription_chunk}""',
                    max_length=30,
                    min_length=10,
                    do_sample=False,
                )
                summaries.append(summary[0]["summary_text"])
                print(f"Current summary is {summary[0]}")
                transcription_chunk = sentence
                tokens_used_so_far = len(sentence.split())

        if transcription_chunk != "":
            summary = summarizer(
                f'""{transcription_chunk}""',
                max_length=30,
                min_length=10,
                do_sample=False,
            )
            summaries.append(summary[0]["summary_text"])
        print(f"List of summaries so far: {summaries}\n")
        concat_summaries = " ".join(summaries)
        print(f"no tokens in concat summary is {len(concat_summaries.split())}\n")
        summary = summarizer(
            f'""{transcription_chunk}""',
            max_length=150,
            min_length=50,
            do_sample=False,
        )
        print(f"When using {sum_model} final summary is:\n {summary}")
    with open(f"data/bart_summaries/experiment.txt", "w") as f:
        f.write(summary[0]["summary_text"])
    return summary[0]["summary_text"]


def generate_summary(str_trans_sp):
    st.write("Input transcription is:", str_trans_sp)
    no_tokens = 0
    for utterance in str_trans_sp:
        no_tokens += len(utterance.split())
    st.write(f"There are {no_tokens} tokens in this transcript separated by spaces")

    # summarize using chosen models
    options = st.multiselect(
        'What model/models do you want to use? Select the label "Done" when you are done',
        options=['', 'BART', 'GPT3', 'Done', 'Blue'])
    if options == '':
        st.write("Script will pause until a valid option is selected")
    elif 'Done' not in options:
        st.write("Script will pause until label 'Done' is also added")
    else:
        st.write(f"Using the following models:  {options}")
        # models = ["BART", "GPT3"]
        models = [option for option in options if option != "Done"]
        # models = ["BART"]
        max_tokens_real = {
            "BART": 1024,
            "GPT3": 2048,
            "DialogueLM": 5120,
            "DialogueLMSparse": 8192,
        }
        max_tokens = {
            "BART": 600,
            "GPT3": 1600,
            "DialogueLM": 4700,
            "DialogueLMSparse": 7600,
        }  # number of tokens is not exactly number of words so using a lower upper limit
        for sum_model in models:
            st.write(f"## Using {sum_model} for summarization")
            assert sum_model in ["BART", "GPT3"], "Model not supported yet"
            max_tokens_model = max_tokens[sum_model]
            if sum_model == "BART":
                using_BART(no_tokens, max_tokens_model, str_trans_sp, sum_model)
            elif sum_model == "GPT3":
                using_GPT3(no_tokens, max_tokens_model, str_trans_sp, sum_model)
            elif sum_model == "Longformer":
                using_Longformer(no_tokens, max_tokens_model, str_trans_sp, sum_model)


def main():
    parser = argparse.ArgumentParser(description="Extracts meeting minutes")
    parser.add_argument(
        "--input_type",
        type=str,
        # default=os.path.join(project_dir, 'data/ami-summary/'),
        default="transcript",
        choices=["transcript", "video", "audio"],
        help="Type of input data. Can be audio, video or transcripts",
    )
    parser.add_argument(
        "--file_path",
        type=Path,
        # default=os.path.join(project_dir, 'data/ami-summary/'),
        # default="/scratch/shared/beegfs/oncescu/shared-datasets/dialogue/test_meeting.mp4",
        default="./data/AMICorpus/ES2008a.transcript.txt",
        help="Location of audio/video/transcript. Currently only takes one file",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        # default=os.path.join(project_dir, 'data/ami-summary/'),
        # default="/scratch/shared/beegfs/oncescu/shared-datasets/dialogue/test_meeting.mp4",
        default=8,
        help="How many sentences per group are being summarised",
    )
    args = parser.parse_args()
    st.title("Meeting summarization")
    input_type = st.selectbox(
        "Set type of input data:",
        key="input_type",
        options=["", "transcript", "video", "audio"],
    )
    if input_type == "":
        st.write("Script will pause until a valid option is selected")
    elif input_type != "transcript":
        # if args.input_type != "transcript":
        st.write("### Am audio file needs to be uploaded. Currently using demo one")
        # extract audio from video if needed since currently only using audio for diarization
        desired_audio_type = "wav"
        file_name = str(args.file_path).rsplit("/", 1)[1].rsplit(".", 1)[0]
        if ".mp4" in str(args.file_path):
            extract_audio(desired_audio_type, args.file_path, file_name)

        # extract speakers and their moments
        pipeline_diar = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
        start_end_times, speakers = dizarization_fct(
            pipeline_diar, args.file_path, file_name, desired_audio_type
        )

        # whisper now to get transcripts
        transcriptions = get_transcriptions(
            start_end_times, args.file_path, file_name, desired_audio_type, speakers
        )

        # get transcriptions in desired str form for future summarization
        str_trans_sp = ""
        for idx, transcription in enumerate(tqdm.tqdm(transcriptions)):
            str_trans_sp += f"{speakers[idx]}: {transcription}\n "
        # print("Transcription is:\n {str_trans_sp}")
        generate_summary(str_trans_sp)
    else:
        st.write("### A transcript needs to be uploaded. Currently using demo one")
        with open(args.file_path, "r") as f:
            str_trans_sp = f.read().splitlines()
        generate_summary(str_trans_sp)


if __name__ == "__main__":
    main()
