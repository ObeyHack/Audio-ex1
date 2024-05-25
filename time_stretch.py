"""
In this file we will experiment with naively interpolating a signal on the time domain and on the frequency domain.

We reccomend you answer this file last.
"""
import torchaudio as ta
import soundfile as sf
import torch
import typing as tp
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
from general_utilities import *
from torch.nn.functional import interpolate


def naive_time_stretch_temporal(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that uses a simple linear interpolation across the temporal dimension
      stretching/squeezing a given waveform by a given factor.
      Use imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """
    wav = wav.unsqueeze(0)
    stretched_waveform = interpolate(wav, scale_factor=factor, mode='linear', align_corners=False)
    stretched_waveform = stretched_waveform.squeeze(0)
    return stretched_waveform


def naive_time_stretch_stft(wav: torch.Tensor, factor: float):
    """
    Q:
      write a function that converts a given waveform to stft, then uses a simple linear interpolation 
      across the temporal dimension stretching/squeezing by a given factor and converts the stretched signal 
      back using istft.
      Use general_utilities for STFT / iSTFT and imported 'interpolate'.

    1. load audio_files/Basta_16k.wav
    2. use this function to stretch it by 1.2 and by 0.8.
    3. save files using ta.save(fpath, stretch_wav, 16000) and listen to the files. What happened?
       Explain what differences you notice and why that happened in your PDF file

    Do NOT include saved audio in your submission.
    """

    stft = do_stft(wav)
    # take the real part of the stft
    stft_real = stft[..., 0]
    stft_complex = stft[..., 1]
    stretched_stft_real = interpolate(stft_real, scale_factor=factor, mode='linear', align_corners=False)
    stretched_stft_complex = interpolate(stft_complex, scale_factor=factor, mode='linear', align_corners=False)
    streched_stft=torch.stack([stretched_stft_real, stretched_stft_complex], dim=-1)
    stretched_waveform = do_istft(streched_stft)
    return stretched_waveform
