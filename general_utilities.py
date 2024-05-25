"""
This file will define the general utility functions you will need for you implementation throughout this ex.
We suggest you start with implementing and testing the functions in this file.

NOTE: each function has expected typing for it's input and output arguments. 
You can assume that no other input types will be given and that shapes etc. will be as described.
Please verify that you return correct shapes and types, failing to do so could impact the grade of the whole ex.

NOTE 2: We STRONGLY encourage you to write down these function by hand and not to use Copilot/ChatGPT/etc.
Implementaiton should be fairly simple and will contribute much to your understanding of the course material.

NOTE 3: You may use external packages for fft/stft, you are requested to implement the functions below to 
standardize shapes and types.
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


def create_single_sin_wave(frequency_in_hz, total_time_in_secs=3, sample_rate=16000):
    timesteps = np.arange(0, total_time_in_secs * sample_rate) / sample_rate
    sig = np.sin(2 * np.pi * frequency_in_hz * timesteps)
    return torch.Tensor(sig).float()


def load_wav(abs_path: tp.Union[str, Path]) -> tp.Tuple[torch.Tensor, int]:
    """
    This function loads an audio file (mp3, wav).
    If you are running on a computer with gpu, make sure the returned objects are mapped on cpu.

    abs_path: path to the audio file (str or Path)
    returns: (waveform, sample_rate)
        waveform: torch.Tensor (float) of shape [1, num_channels]
        sample_rate: int, the corresponding sample rate
    """
    waveform, sample_rate = ta.load(abs_path)
    return waveform.cpu(), sample_rate


def do_stft(wav: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    """
    This function performs STFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.stft.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    n_fft: int, denoting the number of used fft bins.

    returns: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    """
    stft = torch.stft(wav, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, return_complex=True)

    # make real stft
    stft = torch.view_as_real(stft)
    return stft


def do_istft(spec: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    """
    This function performs iSTFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see torch.istft.

    spec: torch.tensor of the shape (1, n_fft, *, 2) or (B, 1, n_fft, *, 2), where last dim stands for real/imag entries.
    n_fft: int, denoting the number of used fft bins.

    returns: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.

    NOTE: you may need to use torch.view_as_complex.
    """

    # Remove the channel dimension for torch.istft
    spec = spec.squeeze(1)  # Shape: (B, n_fft, num_frames, 2)

    # Convert real and imaginary parts to a complex tensor
    spec = torch.view_as_complex(spec)  # Shape: (B, n_fft, num_frames)

    # Perform inverse STFT
    istft = torch.istft(spec, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft)

    return istft


def do_fft(wav: torch.Tensor) -> torch.Tensor:
    """
    This function performs STFT using win_length=n_fft and hop_length=n_fft//4.
    Should return the complex spectrogram.

    hint: see scipy.fft.fft / torch.fft.rfft, you can convert the input tensor to numpy just make sure to cast it back to torch.

    wav: torch tensor of the shape (1, T).

    returns: corresponding FFT transformation considering ONLY POSITIVE frequencies, returned tensor should be of complex dtype.
    """
    fft = torch.fft.fft(wav)
    #print argmax
    #view as real
    #fft = torch.view_as_real(fft)
    #print(fft.argmax())

    return fft


def plot_spectrogram(wav: torch.Tensor, n_fft: int = 1024) -> None:
    """
    This function plots the magnitude spectrogram corresponding to a given waveform.
    The Y axis should include frequencies in Hz and the x axis should include time in seconds.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    """
    stft = do_stft(wav, n_fft=n_fft)
    stft = torch.view_as_complex(stft)
    plt.imshow(torch.abs(stft).cpu().numpy(), aspect='auto', origin='lower')
    plt.show()


def plot_fft(wav: torch.Tensor) -> None:
    """
    This function plots the FFT transform to a given waveform.
    The X axis should include frequencies in Hz.

    NOTE: As abs(FFT) reflects around zero, please plot only the POSITIVE frequencies.

    wav: torch tensor of the shape (1, T) or (B, 1, T) for the batched case.
    """
    # plot wav
    # plot with pandas and not with plt

    plt.plot(torch.abs(do_fft(wav)).cpu().numpy())
    plt.show()
