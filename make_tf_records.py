from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
from io import *
import os.path

path = r".\Dataset\clean"
path_noisy = r".\Dataset\noisy"
#save_path=r'.\Dataset\records.tfrecords'
#out_file = tf.io.TFRecordWriter(save_path)
files_number = len(os.listdir(path))
i=0

def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    overlap = int(window_size * stride)
    slices = []
    for beg_i in range(0, n_samples, overlap):
        end_i = beg_i + window_size
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float64)

def _float32_feature(value):
    """Wrapper for inserting a float32 Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float32_feature_list(values):
    """Wrapper for inserting a float32 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    """
    return tf.train.FeatureList(feature=[_float32_feature(v) for v in values])

def make_tf_seq_example(sequence_1, sequence_2):
    feature_lists = tf.train.FeatureLists(
        feature_list={  # Serial data uses FeatureLists
            "clean": _float32_feature_list(sequence_1),
            "noisy": _float32_feature_list(sequence_2)
        }
    )
    tf_seq_ex = tf.train.SequenceExample(feature_lists=feature_lists)
    return tf_seq_ex

for file in os.listdir(path):  # iterate over each image
    i+=1
    name, _ = os.path.splitext(file)
    fm, wav_data = wavfile.read(os.path.join(path, file)) # read the wav file
    if fm != 16000: # check it sampling
        raise ValueError('Sampling rate is expected to be 16kHz!')
    audio_serial_clean = slice_signal(wav_data, 2 ** 14) # transform it into a np.array
    fm, wav_data = wavfile.read(os.path.join(path_noisy, name + '_CAFE_SNR_0.wav')) # read the wav file
    if fm != 16000: # check it sampling
        raise ValueError('Sampling rate is expected to be 16kHz!')
    audio_serial_noisy = slice_signal(wav_data, 2 ** 14) # transform it into a np.array
    audio_serial_clean = audio_serial_clean/2**14
    audio_serial_noisy = audio_serial_noisy/2**14
    j=1
    for seq_clean, seq_noisy in zip(audio_serial_clean, audio_serial_noisy):
        save_path = './Dataset/record/' + name + f'_{j}_records.tfrecords'
        j+=1
        out_file = tf.io.TFRecordWriter(save_path)
        tfrec = make_tf_seq_example(seq_clean, seq_noisy)
        out_file.write(tfrec.SerializeToString())
        out_file.close()

    prog = (i/files_number)*100
    if i%30 == 0:
        print(f'progress {prog}%.')



