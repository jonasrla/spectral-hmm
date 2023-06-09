import tensorflow as tf
import numpy as np
import python_speech_features as psf
import hmmlearn.hmm as hmm

# Set the TensorFlow random seed for reproducibility
tf.random.set_seed(42)

# Define the parameters for STFT and HMM
n_fft = 1024  # Number of FFT points
hop_length = 256  # Hop length for STFT
n_mfcc = 13  # Number of MFCC coefficients
n_states = 5  # Number of HMM states

# Dictionary of words
word_dict = {
    0: 'apple',
    1: 'banana',
    2: 'orange',
    # Add more words to the dictionary as needed
}

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

# Function to train the HMM for each word
def train_hmm(features, labels):
    models = {}
    for word_id, word in word_dict.items():
        # Select the features and labels for the current word
        word_features = features[labels == word_id]

        # Create and train the HMM model
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
        model.fit(word_features)

        # Add the trained model to the dictionary
        models[word_id] = model

    return models

# Load and process the training data
training_files = [
    'apple1.wav',
    'apple2.wav',
    'banana1.wav',
    'banana2.wav',
    'orange1.wav',
    'orange2.wav',
]

all_features = []
all_labels = []

for file in training_files:
    word_id = int(file.split('.')[0][-1])
    features = extract_features(file)
    all_features.extend(features)
    all_labels.extend([word_id] * len(features))

all_features = np.array(all_features)
all_labels = np.array(all_labels)

# Train the HMM models
hmm_models = train_hmm(all_features, all_labels)

# Example usage: Classify a new word
new_word_file = 'apple3.wav'
new_word_features = extract_features(new_word_file)

best_score = float('-inf')
best_word = None

for word_id, model in hmm_models.items():
    score = model.score(new_word_features)
    if score > best_score:
        best_score = score
        best_word = word_dict[word_id]

print('Best matched word:', best_word)
