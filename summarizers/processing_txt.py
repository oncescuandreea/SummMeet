from summarizers.summary_models import generate_summary
import streamlit as st
import os
from pathlib import Path
from utils import *


def generate_summary_from_txt(args, file_loc):
    with open(file_loc, "r") as f:
        str_trans_sp = f.read().splitlines()
    generate_summary(str_trans_sp, args)


def process_text_and_selection(args,):
    upload_or_not = st.selectbox(
        "Do you want to upload a file, use the default or select from existent?",
        key="upload_or_not",
        options=["", "Upload", "Default", "Select from existent"],
    )
    if upload_or_not == "":
        st.write("Script is waiting for selecting one option")
    elif upload_or_not == "Default":
        st.write("### A transcript needs to be uploaded. Currently using demo one")
        generate_summary_from_txt(args, args.file_path)
    elif upload_or_not == "Upload":
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
                args,
                os.path.join(save_location, uploaded_file.name),
            )
        else:
            st.write("Waiting for file to be uploaded")
    elif upload_or_not == "Select from existent":
        save_location = Path("./data/AMICorpus/")
        full_opt_list = [""]
        existent_videos = os.listdir(save_location)
        existent_videos = [
            video for video in existent_videos if "mp4" or "mkv" in video
        ]
        full_opt_list.extend(existent_videos)
        uploaded_file = st.selectbox(
            "Select from existent files",
            key="select_preexisting",
            options=full_opt_list,
        )
        if uploaded_file != "":
            generate_summary_from_txt(
                args,
                Path(f"./data/AMICorpus/{uploaded_file}"),
            )
        else:
            st.write("Waiting for your selection")