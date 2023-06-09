import tensorflow as tf
import numpy as np
import python_speech_features as psf
import hmmlearn.hmm as hmm
import os

# Set the TensorFlow random seed for reproducibility
tf.random.set_seed(42)

# Define the parameters for STFT and HMM
n_fft = 1024  # Number of FFT points
hop_length = 256  # Hop length for STFT
n_mfcc = 13  # Number of MFCC coefficients
n_states = 5  # Number of HMM states

# Dataset paths
dataset_path = 'data'  # Path to the dataset folder
speaker_folders = [f"0{i}" for i in range(1, 61)]  # Speaker folders numbered from 01 to 60

# Function to extract features from a WAV file
def extract_features(wav_file):
    # Load the WAV file
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_file))
    waveform = tf.squeeze(waveform)

    # Compute STFT
    stft = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop_length)
    spectrogram = tf.abs(stft)

    # Convert spectrogram to mel-scale
    mel_filterbank = psf.get_filterbanks(nfilt=n_mfcc, nfft=n_fft, samplerate=sample_rate)
    mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_filterbank)

    # Compute logarithm of mel-spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCC features
    mfcc = psf.mfcc(log_mel_spectrogram.numpy(), samplerate=sample_rate, numcep=n_mfcc)

    return mfcc

# Function to train the HMM for each number
def train_hmm(features, labels):
    models = {}
    for number in range(11):  # Numbers 0 to 10
        # Select the features and labels for the current number
        number_features = features[labels == number]

        # Create and train the HMM model
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
        model.fit(number_features)

        # Add the trained model to the dictionary
        models[number] = model

    return models

# Function to test and evaluate the HMM models
def test_hmm(models, test_features, test_labels):
    num_examples = len(test_features)
    num_correct = 0

    for i in range(num_examples):
        example_features = test_features[i]
        example_label = test_labels[i]

        best_score = float('-inf')
        best_number = None

        for number, model in models.items():
            score = model.score(example_features)
            if score > best_score:
                best_score = score
                best_number = number

        if best_number == example_label:
            num_correct += 1

    accuracy = num_correct / num_examples
    return accuracy

# Load and process the training data
all_features = []
all_labels = []

for speaker_folder in speaker_folders:
    speaker_path = os.path.join(dataset_path, speaker_folder)
    wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]

    for wav_file in wav_files:
        file_parts = os.path.splitext(wav_file)[0].split('_')
        digit_spoken = int(file_parts[0])
        features = extract_features(os.path.join(speaker_path, wav_file))
        all_features.append(features)
        all_labels.append(digit_spoken)

all_features = np.concatenate(all_features, axis=0)
all_labels = np.array(all_labels)

# Split the data into training and testing sets
test_split = 0.2
num_examples = len(all_features)
num_test = int(test_split * num_examples)
num_train = num_examples - num_test

# Shuffle the data before splitting
shuffle_indices = np.random.permutation(num_examples)
shuffled_features = all_features[shuffle_indices]
shuffled_labels = all_labels[shuffle_indices]

train_features = shuffled_features[:num_train]
train_labels = shuffled_labels[:num_train]
test_features = shuffled_features[num_train:]
test_labels = shuffled_labels[num_train:]

# Train the HMM models
hmm_models = train_hmm(train_features, train_labels)

# Test and evaluate the HMM models
accuracy = test_hmm(hmm_models, test_features, test_labels)
print('Accuracy:', accuracy)
