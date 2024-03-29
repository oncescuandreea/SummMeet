import subprocess

import tqdm
import whisper
from moviepy.editor import *
from pydub import AudioSegment

import youtube_dl
import streamlit as st


def extract_audio(desired_audio_type, file_path, file_name):
    """
    This function extracts either mp3 or wav from a video file.
    desired_audio_type:(str) wav or mp3 currently supported
    file_path:(Path) Path of video from which audio to be extracted
    file_name:(str) Name of video to be extracted. Included in file_path
    """
    assert desired_audio_type in ["mp3", "wav"], "Desired audio type not supported"
    if desired_audio_type == "mp3":
        video = VideoFileClip(str(file_path))
        # if os.path.exists(file_path.parent / f"{file_name}.mp3") is False:
        if os.path.exists(os.path.join(file_path.parent, f"{file_name}.mp3")) is False:
            st.write("Extracting audio from video")
            video.audio.write_audiofile(
                str(os.path.join(file_path.parent, f"{file_name}.mp3"))
            )
        else:
            st.write("Already extracted mp3 version of this video")
    elif desired_audio_type == "wav":
        # if os.path.exists(file_path.parent / f"{file_name}.wav") is False:
        if os.path.exists(os.path.join(file_path.parent, f"{file_name}.wav")) is False:
            st.write("Extracting audio from video")
            command = f"ffmpeg -i {str(file_path)} -ab 160k -ac 2 -ar 44100 -vn {str(os.path.join(file_path.parent, f'{file_name}.wav'))}"
            # st.write(f"{os.path.join(file_path.parent, f'{file_name}.wav')}")
            subprocess.call(command, shell=True)
        else:
            st.write("Already extracted wav version of this video")


def get_transcriptions(
    start_end_times, file_path, file_name, desired_audio_type, speakers
):
    # https://stackoverflow.com/questions/37999150/how-to-split-a-wav-file-into-multiple-wav-files
    model = whisper.load_model("base")
    transcriptions = []
    no_utterances = len(start_end_times)
    total_progress = 0.0
    my_bar = st.progress(total_progress)
    delta_progress = 1.0 / no_utterances
    for idx, start_end_pairs in enumerate(tqdm.tqdm(start_end_times)):
        t1 = start_end_pairs[0] * 1000  # Works in milliseconds
        t2 = start_end_pairs[1] * 1000
        newAudio = AudioSegment.from_wav(
            str(file_path.parent / f"{file_name}.{desired_audio_type}")
        )
        newAudio = newAudio[t1:t2]
        newAudio.export(
            str(file_path.parent / f"{file_name}_0.{desired_audio_type}"), format="wav"
        )  # Exports to a wav file in the current path.
        test_meeting_0_tr = model.transcribe(
            str(file_path.parent / f"{file_name}_0.{desired_audio_type}")
        )
        #     test_meeting_0_tr = model.transcribe("/scratch/shared/beegfs/oncescu/shared-datasets/dialogue/test_meeting_0.wav")
        # print(f"{speakers[idx]}:{test_meeting_0_tr}\n")
        transcriptions.append(test_meeting_0_tr["text"])
        my_bar.progress(min(1.0, total_progress + delta_progress))
        total_progress += delta_progress
    return transcriptions


def download_audio_only(cwd, youtube_id):

    # for video_link in list_video_links:
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [],
        "writeinfojson": True,
        "writesubtitles": True,
        "outtmpl": "./data/media/%(title)s.%(ext)s",
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"http://www.youtube.com/watch?v={youtube_id}"])


def download_full_yb(youtube_id):

    # for video_link in list_video_links:
    ydl_opts = {
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        # "format": "bestvideo[mp4]+bestaudio[ext=m4a]/best[mp4]",
        # "videoformat": "mp4",
        # 'format': 134,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # one of avi, flv, mkv, mp4, ogg, webm
            },
            # {
            #     'key': 'FFmpegExtractAudio',  # what to use for video format converting?
            #     'preferredcodec': 'wav',  # what to use for video format converting?
            # }
        ],
        # 'format': 134,
        # 'outtmpl': "/data/media/%(title)s.%(ext)s",
        "outtmpl": f"./data/media/{youtube_id}.%(ext)s",
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"http://www.youtube.com/watch?v={youtube_id}"])


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