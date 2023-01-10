import argparse
from pathlib import Path


import tqdm
from huggingface_hub import notebook_login

from pyannote.audio import Pipeline

from summarizers.summary_models import *

from rouge import Rouge
from utils import *
import streamlit as st

from io import StringIO
import pandas as pd

from pathlib import Path
from sys import platform


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


def generate_summary(str_trans_sp, args):
    st.write("Input transcription is:", str_trans_sp)
    no_tokens = 0
    for utterance in str_trans_sp:
        no_tokens += len(utterance.split())
    st.write(f"There are {no_tokens} tokens in this transcript separated by spaces")

    # summarize using chosen models
    options = st.multiselect(
        'What model/models do you want to use? Select the label "Done" when you are done',
        options=["", "BART", "GPT3", "Done", "Blue"],
    )
    if options == "":
        st.write("Script will pause until a valid option is selected")
    elif "Done" not in options:
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
                using_GPT3(no_tokens, max_tokens_model, str_trans_sp, sum_model, args)
            elif sum_model == "Longformer":
                using_Longformer(no_tokens, max_tokens_model, str_trans_sp, sum_model)


def generate_summary_from_txt(file_loc, args):
    with open(file_loc, "r") as f:
        str_trans_sp = f.read().splitlines()
    generate_summary(str_trans_sp, args)


def generate_summary_from_audvid(args, file_path, youtube_id=None):
    desired_audio_type = "wav"
    if "win" in platform:
        file_name = str(file_path).rsplit("\\", 1)[1].rsplit(".", 1)[0]
    else:
        file_name = str(file_path).rsplit("/", 1)[1].rsplit(".", 1)[0]
    if ".mp4" in str(file_path) or ".mkv" in str(file_path):
        extract_audio(desired_audio_type, file_path, file_name)

    # extract speakers and their moments
    pipeline_diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=args.diar_auth_key
    )

    st.write("### 1. Now diarizing speech to obtain time intervals for each utterance")
    start_end_times, speakers = dizarization_fct(
        pipeline_diar, file_path, file_name, desired_audio_type
    )

    # whisper now to get transcripts
    st.write("### 2. Please wait while Whisper extracts transcriptions")
    transcriptions = get_transcriptions(
        start_end_times, file_path, file_name, desired_audio_type, speakers
    )

    # get transcriptions in desired str form for future summarization
    str_trans_sp = ""
    no_utterances = len(transcriptions)
    total_progress = 0.0
    my_bar = st.progress(total_progress)
    delta_progress = 1.0 / no_utterances
    st.write("### 3. Now generating txt file containing transcript")
    prev_speaker = speakers[0]
    for idx, transcription in enumerate(tqdm.tqdm(transcriptions)):
        if idx == 0:
            str_trans_sp += f"{speakers[idx]}: {transcription}."
        else:
            if speakers[idx] == prev_speaker:
                str_trans_sp += f" {transcription}."
            else:
                str_trans_sp += f"\n{speakers[idx]}: {transcription}."
                prev_speaker = speakers[idx]
        my_bar.progress(min(1.0, total_progress + delta_progress))
        total_progress += delta_progress
    (Path("./data/transcripts")).mkdir(parents=True, exist_ok=True)
    if youtube_id is not None:
        with open(Path(f"./data/transcripts/{youtube_id}.txt"), "w") as f:
            f.write(str_trans_sp)
    else:
        with open(Path(f"./data/transcripts/your_transcript.txt"), "w") as f:
            f.write(str_trans_sp)
    # print("Transcription is:\n {str_trans_sp}")
    st.write("## Transcript is:")
    st.write(str_trans_sp)
    generate_summary(str_trans_sp, args)


def main():
    parser = argparse.ArgumentParser(description="Extracts meeting minutes")
    parser.add_argument(
        "--file_path",
        type=Path,
        # default="/scratch/shared/beegfs/oncescu/shared-datasets/dialogue/test_meeting.mp4",
        default="./data/AMICorpus/ES2008a.transcript.txt",
        help="Location of audio/video/transcript. Currently only takes one file",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="How many sentences per group are being summarised",
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        help="OpenAI token to be able to use GPT3",
    )
    parser.add_argument(
        "--diar_auth_key",
        type=str,
        help="HuggingFace token to be able to use diarization tool. Need to accept terms and conditions",
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
        upload_or_not = st.selectbox(
            "Do you want to upload a file or use the default?",
            key="upload_or_not",
            options=[
                "",
                "Upload audio/video",
                "Add youtube link",
                "Select from existent",
                "Default",
            ],
        )
        if upload_or_not == "":
            st.write("Script is waiting for selecting something")
        elif upload_or_not == "Default":
            st.write("### Using demo video provided. There's no video provided yet.")
            # extract audio from video if needed since currently only using audio for diarization
            pass
            generate_summary_from_audvid(args, args.file_path)
        elif upload_or_not == "Select from existent":
            full_opt_list = [""]
            existent_videos = os.listdir("./data/media")
            existent_videos = [
                video for video in existent_videos if "mp4" or "mkv" in video
            ]
            full_opt_list.extend(existent_videos)
            uploaded_file = st.selectbox(
                "Select from existent files",
                key="select_preexisting",
                options=full_opt_list,
            )
            if uploaded_file is not "":
                generate_summary_from_audvid(
                    args,
                    Path(f"./data/media/{uploaded_file}"),
                )
            else:
                st.write("Waiting for your selection")
        elif upload_or_not == "Upload audio/video":
            pass
        elif upload_or_not == "Add youtube link":
            youtube_id = st.text_input(
                "Enter id of youtube meeting video  👇",
                label_visibility="visible",
                disabled=False,
                placeholder="This is a placeholder",
            )
            if youtube_id:
                # st.write("You entered: ", youtube_id)
                if os.path.exists(f"./data/media/{youtube_id}.mp4") is True:
                    st.write(f"Already downloaded this video")
                    ext = ".mp4"
                elif os.path.exists(f"./data/media/{youtube_id}.mkv") is True:
                    st.write(f"Already downloaded this video")
                    ext = ".mkv"
                else:
                    save_location = Path("./data/media/")
                    save_location.mkdir(parents=True, exist_ok=True)
                    download_full_yb(youtube_id)
                    st.write("Finished downloading video")
                    ext = ".mp4"
                generate_summary_from_audvid(
                    args, Path(f"./data/media/{youtube_id}{ext}"), youtube_id
                )
            else:
                st.write("Script is waiting for inserting youtube id")
    else:
        upload_or_not = st.selectbox(
            "Do you want to upload a file or use the default?",
            key="upload_or_not",
            options=["", "Upload", "Default", "Select from existent"],
        )
        if upload_or_not == "":
            st.write("Script is waiting for selecting one option")
        elif upload_or_not == "Default":
            st.write("### A transcript needs to be uploaded. Currently using demo one")
            generate_summary_from_txt(args.file_path, args)
        elif upload_or_not in ["Upload", "Select from existent"]:
            uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
            if uploaded_file is not None:
                file_details = {
                    "FileName": uploaded_file.name,
                    "FileType": uploaded_file.type,
                }
                st.write(file_details)
                save_location = Path("./data/AMICorpus/")
                with open(os.path.join(save_location, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Saved File")
                generate_summary_from_txt(
                    os.path.join(save_location, uploaded_file.name), args
                )
            else:
                st.write("Waiting for file to be uploaded")


if __name__ == "__main__":
    main()
