import argparse
from pathlib import Path

from summarizers.processing_vid import process_video_and_selection
from summarizers.processing_txt import process_text_and_selection
from summarizers.processing_aud import process_audio_and_selection

from rouge import Rouge
import streamlit as st

from pathlib import Path


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
    elif input_type == "audio":
        process_audio_and_selection(args)
    elif input_type == "video":
        process_video_and_selection(args)
    else:
        process_text_and_selection(args)


if __name__ == "__main__":
    main()
