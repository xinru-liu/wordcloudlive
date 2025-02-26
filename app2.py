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
import wave
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------------------------------------------------
# 1) Point NLTK to your offline stopwords folder (if needed):
# -------------------------------------------------------------------
# nltk.data.path.append("/path/to/offline_nltk_data")

# -------------------------------------------------------------------
# 2) Load Whisper from local model files (if needed):
#    If you have the .pt files in ~/.cache/whisper or a custom folder,
#    you can specify the model path, or just do "small" if it's cached.
# -------------------------------------------------------------------
whisper_model = whisper.load_model("small")  # or a local path if needed

# -------------------------------------------------------------------
# 3) Load LLaMA from a local folder. Use local_files_only to block net usage.
# -------------------------------------------------------------------
LLAMA_MODEL_PATH = "/path/to/my_llama_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, local_files_only=True)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
keyword_extractor_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.2,
    do_sample=False,
    repetition_penalty=1.2,
    device=0 if device == "cuda" else -1
)

app = Flask(__name__)
CORS(app)

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 10
AUDIO_FILE = "live_audio.wav"

audio_queue = queue.Queue()
word_frequencies = Counter()

def record_audio():
    while True:
        print("Recording audio... Speak now!")
        audio_data = sd.rec(int(DURATION * SAMPLE_RATE),
                            samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=np.int16)
        sd.wait()

        with wave.open(AUDIO_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)

        print("Audio recorded. Processing...")
        time.sleep(1)

def transcribe_and_process():
    global word_frequencies

    while True:
        print("Processing Audio...")
        result = whisper_model.transcribe(AUDIO_FILE)
        transcribed_text = result["text"]
        print("Transcribed Text:", transcribed_text)

        if transcribed_text.strip():
            prompt = f"""
Extract the most important keywords or key phrases from the following text. 
Only return a list of keywords, nothing else.

Text: 
{transcribed_text}

Keywords:
"""
            llama_output = keyword_extractor_pipeline(prompt)
            generated_text = llama_output[0]["generated_text"]

            split_marker = "Keywords:"
            if split_marker in generated_text:
                keywords_part = generated_text.split(split_marker, 1)[1]
            else:
                keywords_part = generated_text

            keywords_lines = keywords_part.split('\n')
            extracted_keywords = []
            for line in keywords_lines:
                line = line.strip()
                line = re.sub(r'^[0-9\-\.\)]+\s*', '', line)
                line = re.sub(r'[^a-zA-Z\s]', '', line).lower()
                if line:
                    extracted_keywords.append(line.strip())

            print("LLaMA-Extracted Keywords:", extracted_keywords)

            # Filter out short words or stopwords
            filtered_keywords = [
                kw for kw in extracted_keywords
                if len(kw) >= 3 and kw not in nltk.corpus.stopwords.words('english')
            ]

            word_frequencies.update(filtered_keywords)
            word_frequencies = Counter(dict(word_frequencies.most_common(50)))
        else:
            print("Warning: No text transcribed.")

        print("Updated accumulated word frequencies:", word_frequencies)
        time.sleep(120)

@app.route("/wordcloud")
def get_wordcloud_data():
    return jsonify(word_frequencies)

if __name__ == "__main__":
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=transcribe_and_process, daemon=True).start()
    app.run(debug=True, port=5001, host="0.0.0.0")
