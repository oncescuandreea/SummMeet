import streamlit as st
import os
from pathlib import Path
from utils import *
from sys import platform
from pyannote.audio import Pipeline
from summarizers.summary_models import generate_summary


def process_aud_and_selection(args):
    pass