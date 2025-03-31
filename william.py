import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:/Users/Shakthipriya/Downloads/best_weights_improved.keras")

# Define the labels (modify according to your model's classes)
emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
)

output = client.audio_classification("sample1.flac", model="Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8")

# Function to record audio
def record_audio(duration=3, sr=22050):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    print("Recording complete!")
    return np.squeeze(audio)

# Function to extract MFCC features (matching model input)
def extract_features(audio, sample_rate=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=15)  # Use 15 MFCCs (matches model)
    
    # Ensure that we always have 400 timesteps
    if mfccs.shape[1] < 400:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, 400 - mfccs.shape[1])), mode='constant')
    else:
        mfccs_padded = mfccs[:, :400]  # Trim if too long
    
    return mfccs_padded.reshape(1, 400, 15)  # Reshape to match model input

# Function to predict emotion with confidence filtering
def predict_emotion(audio):
    features = extract_features(audio)
    prediction = model.predict(features)
    
    # Get highest probability and index
    emotion_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction Probabilities: {prediction}")
    print(f"Confidence Score: {confidence:.2f}")

    # If confidence is low, return "neutral"
    if confidence < 0.5:
        return "neutral"
    
    return emotion_labels[emotion_index]

# Main execution
recorded_audio = record_audio()
detected_emotion = predict_emotion(recorded_audio)

print(f"Detected Emotion: {detected_emotion}")
