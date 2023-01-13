import streamlit as st
import os
from pathlib import Path
from utils import *
from sys import platform
from pyannote.audio import Pipeline
from summarizers.summary_models import generate_summary


def generate_summary_from_aud(args, file_path):
    if "win" in platform:
        file_name, audio_type = str(file_path).rsplit("\\", 1)[1].rsplit(".", 1)
    else:
        file_name, audio_type = str(file_path).rsplit("/", 1)[1].rsplit(".", 1)

    # extract speakers and their moments
    pipeline_diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=args.diar_auth_key
    )

    st.write("### 1. Now diarizing speech to obtain time intervals for each utterance")
    start_end_times, speakers = dizarization_fct(
        pipeline_diar, file_path, file_name, audio_type
    )

    # whisper now to get transcripts
    st.write("### 2. Please wait while Whisper extracts transcriptions")
    transcriptions = get_transcriptions(
        start_end_times, file_path, file_name, audio_type, speakers
    )

    # get transcriptions in desired str form for future summarization
    st.write("### 3. Now generating txt file containing transcript")
    str_trans_sp = ""
    no_utterances = len(transcriptions)
    total_progress = 0.0
    my_bar = st.progress(total_progress)
    delta_progress = 1.0 / no_utterances
    prev_speaker = speakers[0]
    for idx, transcription in enumerate(tqdm.tqdm(transcriptions)):
        if idx == 0:
            # Uncomment line below if you want to use speaker information too
            # str_trans_sp += f"{speakers[idx]}: {transcription}."
            str_trans_sp += f"{transcription}."
        else:
            if speakers[idx] == prev_speaker:
                str_trans_sp += f" {transcription}."
            else:
                # Uncomment line below if you want to use speaker information too
                # str_trans_sp += f"\n{speakers[idx]}: {transcription}."
                str_trans_sp += f"\n{transcription}."
                prev_speaker = speakers[idx]
        my_bar.progress(min(1.0, total_progress + delta_progress))
        total_progress += delta_progress
    (Path("./data/transcripts")).mkdir(parents=True, exist_ok=True)

    with open(Path(f"./data/transcripts/{file_name}.txt"), "w") as f:
        f.write(str_trans_sp)

    # print("Transcription is:\n {str_trans_sp}")
    st.write("## Transcript is:")
    st.write(str_trans_sp)
    generate_summary(str_trans_sp, args)


def process_audio_and_selection(args):
    upload_or_not = st.selectbox(
        "Do you want to upload a file or use existent ones?",
        key="upload_or_not",
        options=[
            "",
            "Upload audio",
            "Select from existent",
        ],
    )
    if upload_or_not == "":
        st.write("Script is waiting for selecting something")
    elif upload_or_not == "Select from existent":
        full_opt_list = [""]
        existent_audios = os.listdir("./data/media")
        existent_audios = [
            audio for audio in existent_audios if any(ext in audio for ext in ('wav', 'mp3'))
        ]
        full_opt_list.extend(existent_audios)
        uploaded_file = st.selectbox(
            "Select from existent files",
            key="select_preexisting",
            options=full_opt_list,
        )
        if uploaded_file != "":
            generate_summary_from_aud(
                args,
                Path(f"./data/media/{uploaded_file}"),
            )
        else:
            st.write("Waiting for your selection")
    elif upload_or_not == "Upload audio":
        uploaded_file = st.file_uploader("Choose a media file", type=["mp3", "wav"])
        if uploaded_file is not None:
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
            }
            st.write(file_details)
            save_location = Path("./data/media/")
            with open(os.path.join(save_location, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Saved File")
            print(os.path.join(save_location, uploaded_file.name))
            generate_summary_from_aud(
                args,
                Path(os.path.join(save_location, uploaded_file.name)),
            )
        else:
            st.write("Waiting for file to be uploaded")
