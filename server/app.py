from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import soundfile as sf
import torch
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import librosa
import logging
import os
import subprocess
import time
from uuid import uuid4

def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], check=True)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

history = []
sample_rate = 16000
max_turns = 5

def transcribe_audio(file_path):
    try:
        audio_input, sr = sf.read(file_path)
        if sr != sample_rate:
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=sample_rate)
        if audio_input.ndim > 1:
            audio_input = np.mean(audio_input, axis=1)
        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

@app.route("/chat", methods=["POST"])
def chat():
    global history
    try:
        logger.info("Request received with files: %s", request.files.keys())
        if "audio" not in request.files:
            logger.error("No audio file in request")
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]
        temp_path = "uploaded_audio_temp"
        audio_file.save(temp_path)

        try:
            convert_to_wav(temp_path, "uploaded_audio.wav")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {str(e)}")
            return jsonify({"error": "Audio format conversion failed"}), 400

        if os.path.exists(temp_path):
            os.remove(temp_path)

        user_input = transcribe_audio("uploaded_audio.wav")
        if not user_input.strip():
            return jsonify({"error": "Empty transcription"}), 400

        history.append(f"User: {user_input}")
        history = history[-max_turns * 2:]

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": "\n".join(history) + "\nAI:", "stream": False}
        )
        response.raise_for_status()
        ai_response = response.json().get("response", "[No response]")

        history.append(f"AI: {ai_response}")

        # Generate unique audio filename
        audio_id = str(uuid4())
        filename = f"reply_{audio_id}.mp3"
        tts = gTTS(text=ai_response, lang='en')
        tts.save(filename)

        # Cleanup old files (older than 5 min)
        for f in os.listdir():
            if f.startswith("reply_") and f.endswith(".mp3"):
                if (time.time() - os.path.getctime(f)) > 300:
                    os.remove(f)

        return jsonify({
            "reply": ai_response,
            "history": history,
            "audio_id": audio_id
        })

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<audio_id>")
def get_audio(audio_id):
    filename = f"reply_{audio_id}.mp3"
    if not os.path.exists(filename):
        return "Audio not found", 404
    return send_file(filename, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(port=8000)
