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


def plot_ffts(signals, signals_name):
    # same amount of rows and columns if possible
    fig, axs = plt.subplots(len(signals), 1, figsize=(10, 10))
    for i in range(len(signals)):
        plt.sca(axs[i])
        axs[i].set_title(f"FFT({signals_name[i]})")
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')
        plot_fft(signals[i])
    # set title
    fig.tight_layout()
    plt.show()


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
    # create 1KHz and 3KHz sine waves
    fs = 16000
    signal_length = 3
    sine_1Khz = create_single_sin_wave(1000, 1, fs).unsqueeze(0)
    sine_3Khz = create_single_sin_wave(3000, 1, fs).unsqueeze(0)
    sine_1Khz_3Khz = sine_1Khz + sine_3Khz

    # plot FFT - 3 subplots
    waves = [sine_1Khz, sine_3Khz, sine_1Khz_3Khz]
    names = ['sine(1Khz)', 'sine(3Khz)', 'sine(1Khz) + sine(3Khz)']
    plot_ffts(waves, names)

    # plot STFT
    n_fft = 1024
    wav = torch.cat([sine_1Khz, sine_3Khz, sine_1Khz_3Khz], dim=-1)
    plot_spectrogram(wav, n_fft=n_fft)
    plt.title('STFT of [sine(1Khz), sine(3Khz), sine(1Khz) + sine(3Khz)]')
    plt.show()


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
    # load all phone_*.wav files
    waves = []
    for i in range(12):
        wave, _ = load_wav(f'audio_files/phone_digits_8k/phone_{i}.wav')
        waves.append(wave)

    # plot FFT - 2 subplots
    plot_ffts(waves[:2], ['phone_0.wav', 'phone_1.wav'])

    # plot STFT
    n_fft = 1024
    wav = torch.cat(waves, dim=-1)
    plot_spectrogram(wav, n_fft=n_fft)
    plt.title('STFT of all phone_*.wav files')
    plt.show()


# --------------------------------------------------------------------------------------------------
#     Part B        Part B        Part B        Part B        Part B        Part B        Part B
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Digit Classifier
# --------------------------------------------------------------------------------------------------

def plot_audios(signals, names):
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
    for i in range(3):
        for j in range(4):
            plt.sca(axs[i, j])
            axs[i, j].set_title(f"FFT({names[i * 4 + j]})")
            axs[i, j].set_xlabel('Frequency (Hz)')
            axs[i, j].set_ylabel('Magnitude')
            axs[i, j].set_xlim(0, 200)
            plot_fft(signals[i * 4 + j])

    # set title
    fig.tight_layout()
    plt.show()

def argmax2(arr):
    arr = np.squeeze(arr)
    arg_max = np.argmax(arr)
    arr_without_max = np.delete(arr, arg_max)
    arg_max2 = np.argmax(arr_without_max)
    return arg_max, arg_max2


def analyse_audios():
    signals = [load_wav(f'audio_files/phone_digits_8k/phone_{i}.wav')[0] for i in range(12)]
    names = [f'phone_{i}.wav' for i in range(12)]
    plot_audios(signals, names)
    ffts = [do_fft(signal) for signal in signals]
    mags = [torch.abs(fft) for fft in ffts]
    max2freqs = [argmax2(mag.cpu().numpy()) for mag in mags]
    # each line max freq, max mag, 2nd max freq, 2nd max mag
    for i in range(12):
        print(f'{i}: max freq: {max2freqs[i][0]}, 2nd max freq: {max2freqs[i][1]}\n')


def arg_maxes2digit(arg_max, arg_max2):
    if 90 < arg_max < 95 and 129 < arg_max2 < 135:
        return 0
    elif 117 < arg_max < 123 and 66 < arg_max2 < 72:
        return 1
    elif 66 < arg_max < 72 and 129 < arg_max2 < 135:
        return 2
    elif 144 < arg_max < 150 and 66 < arg_max2 < 72:
        return 3
    elif 74 < arg_max < 79 and 117 < arg_max2 < 122:
        return 4
    elif 74 < arg_max < 79 and 130 < arg_max2 < 135:
        return 5
    elif 75 < arg_max < 79 and 144 < arg_max2 < 148:
        return 6
    elif 118 < arg_max < 123 and 82 < arg_max2 < 87:
        return 7
    elif 82 < arg_max < 87 and 131 < arg_max2 < 137:
        return 8
    elif 82 < arg_max < 87 and 144 < arg_max2 < 148:
        return 9
    elif 91 < arg_max < 97 and 117 < arg_max2 < 123:
        return 10
    elif 91 < arg_max < 97 and 145 < arg_max2 < 150:
        return 11
    return -1


def classify_single_digit(wav: torch.Tensor) -> int:
    """
    Q:
    Write a RULE-BASED (if - else..) function to classify a given single digit waveform.
    Use ONLY functions from general_utilities file.

    Hint: try plotting the fft of all digits.

    wav: torch tensor of the shape (1, T).

    return: int, digit number
    """

    #analyse_audios()
    fft = do_fft(wav[0])
    mags = np.abs(fft)
    mags = mags[1:mags.shape[0] // 2]
    # # find the 2 maximum values
    arg_max, arg_max2 = argmax2(mags)
    return arg_maxes2digit(arg_max, arg_max2)


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
        if np.all(stft[i:i + 99] == 0):
            zero_padding.append(i)

    stft_parts = []
    for i in range(len(zero_padding)):
        if i == 0:
            stft_parts.append(stft[:zero_padding[i]])
        else:
            stft_parts.append(stft[zero_padding[i - 1]:zero_padding[i]])
    if len(zero_padding) == 0:
        stft_parts.append(stft)

    return stft_parts


def concatenated2waves(wav):
    # tuples of start and end indices of each wave
    indices = []
    wav = wav[0]
    # You can assume that there will be at least 100ms of zero padding between digits
    i = 0
    start = 0
    while i < len(wav):
        if torch.all(wav[i:i + 99] == 0):
            end = i
            indices.append((start, end))

            # move i to the next non zero value
            i += 100
            while i < len(wav) and wav[i] == 0:
                i += 1

            start = i

        i += 1

    end = len(wav)
    indices.append((start, end))
    return indices


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
    indices = concatenated2waves(wav)
    waves = [wav[0][start:end].unsqueeze(0) for start, end in indices]
    digits = []
    for wave in waves:
        digit = classify_single_digit(wave)
        digits.append(digit)
    return digits



    # digits = []
    # stft_parts = cut_stft(wav)
    # for part in stft_parts:
    #     arg_max, arg_max2 = argmax2(part)
    #     digit = arg_maxes2digit(arg_max, arg_max2)
    #     digits.append(digit)
    # return digits
