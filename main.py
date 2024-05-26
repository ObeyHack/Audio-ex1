# This is a sample Python script.
import torch
import torchaudio as ta
from matplotlib import pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import digit_classifier
from general_utilities import load_wav
import time_stretch


def stft_stretch():
    strached_stft = time_stretch.naive_time_stretch_stft(load_wav('audio_files\\Basta_16k.wav')[0], 1.2)
    ta.save('audio_files\\Basta_16k_stft12.wav', strached_stft, 16000)

    strached_stft = time_stretch.naive_time_stretch_stft(load_wav('audio_files\\Basta_16k.wav')[0], 0.8)
    ta.save('audio_files\\Basta_16k_stft08.wav', strached_stft, 16000)


def sound_stretch():
    strached_12 = time_stretch.naive_time_stretch_temporal(load_wav('audio_files\\Basta_16k.wav')[0], 1.2)
    strached_08 = time_stretch.naive_time_stretch_temporal(load_wav('audio_files\\Basta_16k.wav')[0], 0.8)
    ta.save('audio_files\\Basta_16k_12.wav', strached_12, 16000)
    ta.save('audio_files\\Basta_16k_08.wav', strached_08, 16000)


def check():
    #Let x(t) = sin(2π · 1000 · t) + sin(2π · 5000 · t). We sample x(t) with sampling rate of 8[KHz].
    #What frequencies would appear in the Fourier transform of the discrete measured signal? Assume ∀|ω| > 2π·5000 :
    #XF (ω) = 0.
    #We’ve seen that in order to be able to reconstruct a certain frequency - we MUST have 2 measured
    #points per cycle. Due to that, it must hold that all frequencies > 1
    #2 · fs are 0 as we cannot reconstruct them. Think
    #which frequencies may appear withing the range we are able to reconstruct.
    #Hints: (i) use what you proved in section 1.2.3; (ii) think what frequencies would have XF (·) ̸= 0. Meaning
    #consider the values where ω and n equals the signal frequencies; (iii) recall that Fourier transform is symmetric
    #around zero (we also have negative frequencies).
    sin_1Khz = digit_classifier.create_single_sin_wave(1000, 1, 8000)
    sin_5Khz = digit_classifier.create_single_sin_wave(5000, 1, 8000)
    sin_1Khz_5Khz = sin_1Khz + sin_5Khz
    digit_classifier.plot_fft(sin_1Khz_5Khz)
    plt.show()



if __name__ == '__main__':
    #check() #works
    #digit_classifier.self_check_fft_stft() #works
    #digit_classifier.audio_check_fft_stft()#works
    #sound_stretch()
    #stft_stretch()

    # digit classification
    # for i in range(12):
    #     digit = digit_classifier.classify_single_digit(load_wav(f'audio_files\\phone_digits_8k\\phone_{i}.wav')[0])
    #     print(f"input: phone_{i}.wav, output: {digit}")

    # digit stream classification
    signals = [signal for signal in [load_wav(f'audio_files\\phone_digits_8k\\phone_{i}.wav')[0] for i in range(12)]]
    # concat signals with least 100ms of zero padding between them
    con_signal = signals[0]
    for i in range(1, 12):
        # pad by exactly 100
        con_signal = torch.nn.functional.pad(con_signal, (0, 100))
        con_signal = torch.cat((con_signal, signals[i]), dim=-1)

    digit_stream = digit_classifier.classify_digit_stream(con_signal)
    print(f"input: phone_0.wav to phone_11.wav, output: {digit_stream}")




