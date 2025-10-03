from fileinput import filename
from flask import Flask, request, jsonify, send_file, render_template_string, Response
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
import threading
import queue
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], check=True)

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use CPU for better compatibility in cloud environments
device = torch.device("cpu")

# Load models lazily to avoid startup delays
processor = None
model = None
models_loading = False
models_loaded = False

def load_models():
    global processor, model, models_loading, models_loaded
    if models_loaded:
        return True
    
    if models_loading:
        # Wait for another thread to finish loading
        while models_loading:
            time.sleep(0.1)
        return models_loaded
    
    models_loading = True
    try:
        logger.info("Loading Whisper models...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
        models_loaded = True
        logger.info("Models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False
    finally:
        models_loading = False

history = []
sample_rate = 16000
max_turns = 5

def transcribe_audio(file_path):
    try:
        if not load_models():
            raise Exception("Failed to load Whisper models")
        
        logger.info("Starting audio transcription...")
        audio_input, sr = sf.read(file_path)
        
        # Limit audio length to prevent timeout (max 30 seconds)
        max_samples = sample_rate * 30
        if len(audio_input) > max_samples:
            audio_input = audio_input[:max_samples]
            logger.info("Audio truncated to 30 seconds")
        
        if sr != sample_rate:
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=sample_rate)
        if audio_input.ndim > 1:
            audio_input = np.mean(audio_input, axis=1)
        
        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        
        with torch.no_grad():
            # Use faster generation parameters
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,  # Faster than beam search
                do_sample=False,
                early_stopping=True
            )
        
        result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info("Transcription completed successfully")
        return result
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

        
        ollama_base_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate").replace("/api/generate", "")
        ollama_api_url = f"{ollama_base_url}/api/generate"
        logger.info(f"Using Ollama URL: {ollama_api_url}")
        
        # Try different model names that might be available (prioritize smaller models)
        model_names = ["llama3.2:1b", "llama3.2", "llama3", "llama2"]
        
        ai_response = "I'm sorry, I'm having trouble connecting to the AI service right now. Please try again in a moment."
        
        for model_name in model_names:
            try:
                logger.info(f"Trying model: {model_name}")
                
                # Add timeout and optimize request
                response = requests.post(
                    ollama_api_url, 
                    json={
                        "model": model_name, 
                        "prompt": "\n".join(history) + "\nAI:", 
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 150  # Limit response length for faster generation
                        }
                    },
                    timeout=30  # 30 second timeout
                )
                response.raise_for_status()
                ai_response = response.json().get("response", "[No response]")
                logger.info(f"Success with model: {model_name}")
                break
            except requests.exceptions.Timeout:
                logger.warning(f"Model {model_name} timed out after 30 seconds")
                continue
            except Exception as e:
                logger.info(f"Model {model_name} failed: {str(e)}")
                continue

        history.append(f"AI: {ai_response}")

        # Generate unique audio filename
        audio_id = str(uuid4())
        filename = f"reply_{audio_id}.mp3"
        full_audio_path = os.path.join(BASE_DIR, filename)
        
        try:
            logger.info("Generating audio response...")
            # Limit text length for faster TTS generation
            tts_text = ai_response[:500] if len(ai_response) > 500 else ai_response
            tts = gTTS(text=tts_text, lang='en', slow=False)
            tts.save(full_audio_path)
            logger.info("Audio generation completed")
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            # Continue without audio if TTS fails

        # Clean up old audio files (run in background)
        def cleanup_old_files():
            try:
                for f in os.listdir(BASE_DIR):
                    if f.startswith("reply_") and f.endswith(".mp3"):
                        full_path = os.path.join(BASE_DIR, f)
                        if (time.time() - os.path.getctime(full_path)) > 300:
                            os.remove(full_path)
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
        
        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=cleanup_old_files)
        cleanup_thread.daemon = True
        cleanup_thread.start()


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
    full_audio_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_audio_path):
        return "Audio not found", 404
    return send_file(full_audio_path, mimetype="audio/mpeg")

@app.route("/status")
def get_status():
    """Return current processing status"""
    return jsonify({
        "models_loaded": models_loaded,
        "models_loading": models_loading,
        "status": "ready" if models_loaded else "loading_models"
    })

@app.route("/health")
def health():
    return {"status": "healthy", "message": "App is running"}, 200

@app.route("/")
def index():
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ GOW'S CHAT PAL ✨</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
        }
        
        .container {
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
            max-width: 600px;
            width: 90%;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .chat-window {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .bot-avatar {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .chat-response {
            font-size: 1.2rem;
            line-height: 1.6;
            text-align: center;
        }
        
        .btn-wrapper {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 1rem 2rem;
            font-size: 1.1rem;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            min-width: 150px;
        }
        
        .btn.start {
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            color: white;
        }
        
        .btn.stop {
            background: linear-gradient(45deg, #ff7675, #fd79a8);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-indicator {
            margin-top: 1rem;
            font-size: 1rem;
            min-height: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ GOW'S CHAT PAL ✨</h1>
        <div class="chat-window">
            <div class="bot-avatar">🌈</div>
            <p class="chat-response" id="response">✨ Ready to chat with you!</p>
        </div>
        <div class="btn-wrapper">
            <button class="btn start" id="startBtn" onclick="handleStart()">
                🎙️ Start Talking
            </button>
            <button class="btn stop" id="stopBtn" onclick="handleStop()" disabled>
                ⛔ Stop
            </button>
        </div>
        <div class="status-indicator" id="status"></div>
        <audio id="audioPlayer" style="display: none;"></audio>
    </div>

    <script>
        let isRecording = false;
        let mediaRecorder = null;
        let isPlaying = false;

        const responseEl = document.getElementById('response');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusEl = document.getElementById('status');
        const audioPlayer = document.getElementById('audioPlayer');

        async function handleStart() {
            isRecording = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            responseEl.textContent = "🎧 Listening... tell me everything!";
            
            try {
                const audioBlob = await startRecording();
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.wav");

                // Show processing status
                responseEl.textContent = "🔄 Processing your message...";
                statusEl.textContent = "Transcribing audio...";

                const result = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });

                if (!result.ok) {
                    throw new Error(`Server error: ${result.status}`);
                }

                const data = await result.json();

                if (data.error) {
                    throw new Error(data.error);
                }
                
                responseEl.textContent = data.reply;
                statusEl.textContent = "Generating audio response...";

                // Play the audio reply
                const audioUrl = `/audio/${data.audio_id}?t=${Date.now()}`;
                audioPlayer.src = audioUrl;
                audioPlayer.play();
                
            } catch (error) {
                console.error('Error:', error);
                responseEl.textContent = `❗ ${error.message || "Something went wrong"}`;
                statusEl.textContent = "Error occurred. Please try again.";
            } finally {
                isRecording = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }

        function handleStop() {
            if (mediaRecorder) {
                mediaRecorder.stop();
            }
            isRecording = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            responseEl.textContent = "🛑 Stopped listening. I'm still here for you 💖";
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
        }

        function startRecording() {
            return new Promise(async (resolve) => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const recorder = new MediaRecorder(stream);
                const audioChunks = [];

                recorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                recorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                    resolve(audioBlob);
                };

                mediaRecorder = recorder;
                recorder.start();
            });
        }

        audioPlayer.onplay = () => {
            statusEl.textContent = "🔊 Playing response...";
        };

        audioPlayer.onended = () => {
            statusEl.textContent = "";
        };
    </script>
</body>
</html>
    """
    return render_template_string(html_template)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

