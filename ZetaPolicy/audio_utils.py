import librosa
import numpy as np


def add_missing_padding(audio, sr, duration):
    signal_length = duration * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def split_audio(signal, sr, split_duration):
    length = split_duration * sr

    if length < len(signal):
        frames = librosa.util.frame(signal, frame_length=length, hop_length=length).T
        return frames
    else:
        audio = add_missing_padding(signal, sr, split_duration)
        frames = [audio]
        return np.array(frames)
