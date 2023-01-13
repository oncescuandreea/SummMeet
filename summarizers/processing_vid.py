import streamlit as st
import os
from pathlib import Path
from utils import *
from sys import platform
from pyannote.audio import Pipeline
from summarizers.summary_models import generate_summary
import tqdm


def generate_summary_from_vid(args, file_path, youtube_id=None):
    desired_audio_type = "wav"
    if "win" in platform:
        file_name = str(file_path).rsplit("\\", 1)[1].rsplit(".", 1)[0]
    else:
        file_name = str(file_path).rsplit("/", 1)[1].rsplit(".", 1)[0]
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
    if youtube_id is not None:
        with open(Path(f"./data/transcripts/{youtube_id}.txt"), "w") as f:
            f.write(str_trans_sp)
    else:
        with open(Path(f"./data/transcripts/{file_name}.txt"), "w") as f:
            f.write(str_trans_sp)
    # print("Transcription is:\n {str_trans_sp}")
    # st.write("## Transcript is:")
    # st.write(str_trans_sp)
    generate_summary(str_trans_sp, args)


def process_video_and_selection(args,):
    upload_or_not = st.selectbox(
        "Do you want to upload a file, download a meeting video from youtube or use existent files?",
        key="upload_or_not",
        options=[
            "",
            "Upload video",
            "Add youtube link",
            "Select from existent",
        ],
    )
    if upload_or_not == "":
        st.write("Script is waiting for selecting something")
    elif upload_or_not == "Select from existent":
        full_opt_list = [""]
        existent_videos = os.listdir("./data/media")
        existent_videos = [
            video for video in existent_videos if any(ext in video for ext in ('mp4', 'mkv'))
        ]
        full_opt_list.extend(existent_videos)
        uploaded_file = st.selectbox(
            "Select from existent files",
            key="select_preexisting",
            options=full_opt_list,
        )
        if uploaded_file != "":
            generate_summary_from_vid(
                args,
                Path(f"./data/media/{uploaded_file}"),
            )
        else:
            st.write("Waiting for your selection")
    elif upload_or_not == "Upload video":
        uploaded_file = st.file_uploader("Choose a media file", type=["mp4", "mkv"])
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
            generate_summary_from_vid(
                args,
                Path(os.path.join(save_location, uploaded_file.name)),
            )
        else:
            st.write("Waiting for file to be uploaded")
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
            generate_summary_from_vid(
                args, Path(f"./data/media/{youtube_id}{ext}"), youtube_id
            )
        else:
            st.write("Script is waiting for inserting youtube id")