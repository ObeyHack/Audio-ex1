"""
This file will implement a digit classifier using rule-based dsp methods.
As all digit waveforms are given, we could take that under consideration, of our RULE-BASED system.

We reccomend you answer this after filling all functions in general_utilities.
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


# --------------------------------------------------------------------------------------------------
#     Part A        Part A        Part A        Part A        Part A        Part A        Part A
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# In this part we will get familiarized with the basic utilities defined in general_utilities
# --------------------------------------------------------------------------------------------------


def self_check_fft_stft():
    """
    Q:
    1. create 1KHz and 3Khz sine waves, each of 3 seconds length with a sample rate of 16KHz.
    2. In a single plot (3 subplots), plot (i) FFT(sine(1Khz)) (ii) FFT(sine(3Khz)),
       (iii) FFT(sine(1Khz) + sine(3Khz)), make sure X axis shows frequencies.
       Use general_utilities.plot_fft
    3. concatate [sine(1Khz), sine(3Khz), sine(1Khz) + sine(3Khz)] along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    # 1.
    sine_1Khz = create_single_sin_wave(1000, 3, 16000)
    sine_3Khz = create_single_sin_wave(3000, 3, 16000)
    sine_1Khz_3Khz = sine_1Khz + sine_3Khz
    # 2.
    plot_fft(do_fft(sine_1Khz))
    plot_fft(do_fft(sine_3Khz))
    plot_fft(do_fft(sine_1Khz_3Khz))
    # 3.
    stft = do_stft(torch.cat([sine_1Khz, sine_3Khz, sine_1Khz_3Khz], dim=-1), n_fft=1024)
    # plot spectrogram
    plot_spectrogram(stft)


def audio_check_fft_stft():
    """
    Q:
    1. load all phone_*.wav files in increasing order (0 to 11)
    2. In a single plot (2 subplots), plot (i) FFT(phone_1.wav) (ii) FFT(phone_2.wav).
       Use general_utilities.plot_fft
    3. concatate all phone_*.wav files in increasing order (0 to 11) along the temporal axis, and plot
       the corresponding MAGNITUDE STFT using n_fft=1024. Make sure Y ticks are frequencies and X
       ticks are seconds.

    Include all plots in your PDF
    """
    # 1.
    phone_wavs = [load_wav(
        f'C:\\Users\\merzi\\PycharmProjects\\pythonProject\\Audioex1\\audio_files\\phone_digits_8k\\phone_{i}.wav')[0]
                  for i in range(12)]
    # make torch.Tensor(wav) of phone_wavs
    phone_wavs = [phone_wav[0] for phone_wav in phone_wavs]
    # 2.
    plot_fft(do_fft(phone_wavs[1]))
    plot_fft(do_fft(phone_wavs[2]))
    # 3.
    stft = do_stft(torch.cat(phone_wavs, dim=-1), n_fft=1024)
    # plot spectrogram
    plot_spectrogram(stft, n_fft=1024)


# --------------------------------------------------------------------------------------------------
#     Part B        Part B        Part B        Part B        Part B        Part B        Part B
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Digit Classifier
# --------------------------------------------------------------------------------------------------

def classify_single_digit(wav: torch.Tensor) -> int:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a given single digit waveform.
    Use ONLY functions from general_utilities file.

    Hint: try plotting the fft of all digits.

    wav: torch tensor of the shape (1, T).

    return: int, digit number
    """
    # plot the fft of all digits
    wav = wav[0]
    plot_fft(do_fft(wav))
    # classify the digit
    fft = do_fft(wav)
    fft = fft.cpu().numpy()
    fft = np.abs(fft)
    # fft = fft[0]
    fft = fft[1:fft.shape[0] // 2]
    # find the 2 maximum values
    arg_max = np.argmax(fft)
    fft_without_max = np.delete(fft, arg_max)
    arg_max2 = np.argmax(fft_without_max)
    # print (arg_max, arg_max2)


    if 90 < arg_max < 95 and 129 < arg_max2 < 135:
        return 0
    elif 117 < arg_max < 123 and 66 < arg_max2 < 72:
        return 1
    elif 66 < arg_max < 72 and 129 < arg_max2 < 135:
        return 2
    elif 144 < arg_max < 150 and 66 < arg_max2 < 72:
        return 3
    elif 75 < arg_max < 80 and 118 < arg_max2 < 124:
        return 4
    elif 75 < arg_max < 80 and 131 < arg_max2 < 137:
        return 5
    elif 75 < arg_max < 80 and 145 < arg_max2 < 151:
        return 6
    elif 119 < arg_max < 125 and 83 < arg_max2 < 89:
        return 7
    elif 83 < arg_max < 89 and 131 < arg_max2 < 137:
        return 8
    elif 83 < arg_max < 89 and 145 < arg_max2 < 151:
        return 9
    return -1


def cut_stft(wav: torch.Tensor):
    # cut the stft to parts by the zero padding
    stft = do_stft(wav, n_fft=1024)
    stft = stft.cpu().numpy()
    stft = np.abs(stft)
    stft = stft[0]
    stft = stft[1:stft.shape[0] // 2]
    # find the zero padding
    zero_padding = []
    for i in range(len(stft)):
        if stft[i:i + 99] == 0:
            zero_padding.append(i)

    stft_parts = []
    for i in range(len(zero_padding)):
        if i == 0:
            stft_parts.append(stft[:zero_padding[i]])
        else:
            stft_parts.append(stft[zero_padding[i - 1]:zero_padding[i]])

    return stft_parts


def classify_digit_stream(wav: torch.Tensor) -> tp.List[int]:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a waveform containing several digit stream.
    The input waveform will include at least a single digit in it.
    The input waveform will have digits waveforms concatenated on the temporal axis, with random zero
    padding in-between digits.
    You can assume that there will be at least 100ms of zero padding between digits
    The function should return a list of all integers pressed (in order).

    Use STFT from general_utilities file to answer this question.

    wav: torch tensor of the shape (1, T).

    return: List[int], all integers pressed (in order).
    """
    # plot the fft of the input waveform
    plot_fft(do_fft(wav))
    # classify the digit stream
    stft = do_stft(wav, n_fft=1024)
    stft = stft.cpu().numpy()
    stft = np.abs(stft)
    stft = stft[0]
    stft = stft[1:stft.shape[0] // 2]
    digits = []
    stft_parts = cut_stft(wav)
    for part in stft_parts:
        # find the 2 maximum values
        arg_max = np.argmax(part)
        part_without_max = np.delete(part, arg_max)
        arg_max2 = np.argmax(part_without_max)
        if 90 < arg_max < 95 and 129 < arg_max2 < 135:
            digits.append(0)
        elif 117 < arg_max < 123 and 66 < arg_max2 < 72:
            digits.append(1)
        elif 66 < arg_max < 72 and 129 < arg_max2 < 135:
            digits.append(2)
        elif 144 < arg_max < 150 and 66 < arg_max2 < 72:
            digits.append(3)
        elif 75 < arg_max < 80 and 118 < arg_max2 < 124:
            digits.append(4)
        elif 75 < arg_max < 80 and 131 < arg_max2 < 137:
            digits.append(5)
        elif 75 < arg_max < 80 and 145 < arg_max2 < 151:
            digits.append(6)
        elif 119 < arg_max < 125 and 83 < arg_max2 < 89:
            digits.append(7)
        elif 83 < arg_max < 89 and 131 < arg_max2 < 137:
            digits.append(8)
        elif 83 < arg_max < 89 and 145 < arg_max2 < 151:
            digits.append(9)

        return digits
