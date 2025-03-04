import sounddevice as sd
import numpy as np
import queue
import whisper
import nltk
import time
import threading
import json
from flask import Flask, jsonify
from flask_cors import CORS
from collections import Counter
from nltk.corpus import stopwords
import wave
import re  # Add this line

import os
import imageio_ffmpeg

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ["PATH"]


# Download NLTK stopwords
nltk.download('stopwords')

# Load Whisper Model
model = whisper.load_model("small")

# Flask App
app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all requests
CORS(app)  # Enable CORS for all routes

# Audio Settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
CHANNELS = 1
DURATION = 10  # Capture every 10 seconds
AUDIO_FILE = "live_audio.wav"

# Queue for live audio processing
audio_queue = queue.Queue()
word_frequencies = Counter()  # Use a Counter to accumulate word counts

# Function to record live audio
def record_audio():
    while True:
        print("Recording audio... Speak now!")

        # Record audio in chunks and save it
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
        sd.wait()

        # Save as a proper WAV file
        with wave.open(AUDIO_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)

        print("Audio recorded. Processing...")
        time.sleep(1)  # Small delay before next recording


# Function to transcribe audio and update word frequencies
def transcribe_and_process():
    global word_frequencies

    while True:
        print("Processing Audio...")
        result = model.transcribe(AUDIO_FILE)
        transcribed_text = result["text"]

        print("Transcribed Text:", transcribed_text)  # Debugging

        if not transcribed_text.strip():
            print("Warning: No text transcribed.")
            # word_frequencies = {}
        else:
            # Keep only alphabetic words (filter out special characters and garbage)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', transcribed_text.lower())

            # Remove stopwords
            filtered_words = [word for word in words if word not in stopwords.words('english')]
            
            # Count word frequencies from the new recording
            new_word_counts = Counter(filtered_words)

            # Accumulate frequencies over time
            word_frequencies.update(new_word_counts)

            # Keep only the top 20 most frequent words
            word_frequencies = Counter(dict(word_frequencies.most_common(50)))

        print("Updated accumulated word frequencies:", word_frequencies)
        time.sleep(120)  # Refresh every 2 minutes

# Flask API to serve word frequencies
@app.route("/wordcloud")
def get_wordcloud_data():
    return jsonify(word_frequencies)

# Start threads
threading.Thread(target=record_audio, daemon=True).start()
threading.Thread(target=transcribe_and_process, daemon=True).start()

# Run Flask server
if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
