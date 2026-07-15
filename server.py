import os
import pickle
import shutil
import tempfile
import torch
import soundfile as sf
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference.speaker import EncoderClassifier

# Replacing transformers with a cleaner, highly-optimized local engine
from faster_whisper import WhisperModel

app = FastAPI()

# Enable CORS for local cross-origin file communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize models
print("Loading Speaker Recognition model...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
DB_PATH = "speaker_database.pkl"

print("Loading Speech-to-Text model (Faster-Whisper)...")
# "tiny" or "base" run perfectly and instantly on local CPUs
asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")

def get_embedding(file_path):
    """Loads an audio file and extracts its clean embedding vector."""
    signal, fs = sf.read(file_path)
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    if len(signal_tensor.shape) > 1:
        signal_tensor = torch.mean(signal_tensor, dim=1)
    signal_tensor = signal_tensor.unsqueeze(0)
    
    with torch.no_grad():
        emb = classifier.encode_batch(signal_tensor)
    return emb.squeeze(1)


@app.post("/enroll")
async def enroll_speaker(username: str = Form(...), file: UploadFile = File(...)):
    """Extracts embedding and saves it to the database file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        print(f"Extracting embedding for {username}...")
        embedding = get_embedding(temp_path)
        os.remove(temp_path)
        
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                database = pickle.load(f)
        else:
            database = {}
            
        database[username] = embedding
        
        with open(DB_PATH, "wb") as f:
            pickle.dump(database, f)
            
        return {"status": "success", "message": f"Successfully enrolled '{username}'"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_and_transcribe(file: UploadFile = File(...), threshold: float = 0.35):
    """Verifies speaker and returns transcription only if verified."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # 1. Check Voice Database
        if not os.path.exists(DB_PATH):
            os.remove(temp_path)
            return {"verified": False, "user": "Unknown", "message": "Database is empty. Please enroll users first."}

        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)

        unknown_emb = get_embedding(temp_path)
        best_score = -1.0
        matched_user = "Unknown / Unauthorized"

        for username, saved_emb in database.items():
            similarity = F.cosine_similarity(unknown_emb, saved_emb).item()
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                matched_user = username

        # 2. Check Authorization
        if matched_user == "Unknown / Unauthorized":
            os.remove(temp_path)
            return {
                "verified": False,
                "user": matched_user,
                "text": "",
                "message": "Access Denied: Voice not recognized."
            }

        # 3. Authorized -> Transcribe Speech
        print(f"Verification successful for {matched_user}. Transcribing command...")
        segments, info = asr_model.transcribe(temp_path, beam_size=5)
        
        # Combine segments into a clean text block
        transcribed_text = " ".join([segment.text for segment in segments]).strip()

        os.remove(temp_path)

        return {
            "verified": True,
            "user": matched_user,
            "text": transcribed_text,
            "score": best_score,
            "message": "Authorized successfully!"
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)