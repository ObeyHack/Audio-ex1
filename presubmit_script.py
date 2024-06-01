import digit_classifier
from digit_classifier import *
from time_stretch import *
from general_utilities import *
import torch
import os
from pathlib import Path
import argparse

def test_classify_single_digit(root):
    wav, _ = load_wav(f'{root}/audio_files/digit_5.wav')
    digit = classify_single_digit(wav)
    assert isinstance(digit, int), "return type should be int"

def test_classify_digit_stream(root):
    wav, _ = ta.load(f'{root}/audio_files/digit_5.wav')
    digit_stream = classify_digit_stream(wav)
    assert isinstance(digit_stream, list) and len(digit_stream) > 0 and isinstance(digit_stream[0], int), "return type should be list of ints"

def test_naive_time_stretch_temporal(root):
    wav, _ = ta.load(f'{root}/audio_files/Basta_16k.wav')
    stretched_wav = naive_time_stretch_temporal(wav, 1.2)
    assert isinstance(stretched_wav, torch.Tensor), "return type should be a torch tensor"

def test_naive_time_stretch_stft(root):
    wav, _ = ta.load(f'{root}/audio_files/Basta_16k.wav')
    stretched_wav = naive_time_stretch_stft(wav, 1.2)
    assert isinstance(stretched_wav, torch.Tensor), "return type should be a torch tensor"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=str(Path(os.path.abspath(__file__)).parent))
    root = parser.parse_args().root
    cur_dir = Path(os.path.abspath(__file__)).parent

    # ---- uid ----
    assert os.path.exists(f"{cur_dir}/ids.txt"), "Missing ID file"
    try:
        with open(f"{cur_dir}/ids.txt", "r") as fp:
            lines = "".join(fp.readlines())
        ids = (lines.split("\n")[0]).split(',')
        ids = [int(uid) for uid in ids]
    except:
        raise Exception("Invalid ID file, should contain a single line following: <id>,<id> format")

    assert os.path.exists(f"{cur_dir}/digit_classifier.py"), "Missing digit_classifier.py file"
    assert os.path.exists(f"{cur_dir}/general_utilities.py"), "Missing general_utilities.py file"
    assert os.path.exists(f"{cur_dir}/time_stretch.py"), "Missing time_stretch.py file"
    
    # ---- Utils ----
    ret = load_wav(f"{root}/audio_files/Basta_16k.wav")
    assert len(ret) == 2, "load_wav should return (wav, sr) tuple"
    assert isinstance(ret[0], torch.Tensor) and ret[0].dtype == torch.float32 and isinstance(ret[1], int), "load_wav should return (torch.Tensor (float), int)"

    dummy = torch.randn(1, 32000)
    ret = do_stft(dummy, n_fft=512)
    assert isinstance(ret, torch.Tensor) and ret.shape[-1] == 2, "do_stft should return torch.Tensor with expected shape"

    ret = do_fft(dummy)
    assert isinstance(ret, torch.Tensor) and ret.dtype in {torch.complex32, torch.complex64}, "do_stft should return torch.Tensor with expected type"

    dummy = torch.randn((257, 32000 // (512 // 4) + 1, 2))
    ret = do_istft(dummy, n_fft=512)
    assert isinstance(ret, torch.Tensor) and ret.shape[-1] == 32000, "do_istft should return torch.Tensor with expected shapes"


    # ---- Digit Classification ----
    test_classify_single_digit(root)
    test_classify_digit_stream(root)

    # ---- Time Stretch ----
    test_naive_time_stretch_temporal(root)
    test_naive_time_stretch_stft(root)





