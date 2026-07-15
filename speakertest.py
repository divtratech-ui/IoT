import os
import pickle
import torch
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

# Initialize model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
DB_PATH = "speaker_database.pkl"

def get_embedding(file_path):
    """Loads an audio file and extracts its clean embedding vector."""
    signal, fs = sf.read(file_path)
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    if len(signal_tensor.shape) > 1:
        signal_tensor = torch.mean(signal_tensor, dim=1)
    signal_tensor = signal_tensor.unsqueeze(0)
    
    with torch.no_grad():
        emb = classifier.encode_batch(signal_tensor)
    return emb.squeeze(1) # Shape: (1, 192)

def enroll_speaker(username, audio_path):
    """Extracts embedding and saves it to the database file."""
    print(f"Extracting embedding for {username}...")
    embedding = get_embedding(audio_path)
    
    # Load existing database or create a new dictionary
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = {}
        
    # Save/Overwrite user
    database[username] = embedding
    
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    print(f"Successfully enrolled '{username}' into the database!")

# --- ENROLLMENT EXAMPLE ---
# Change these paths to add real speakers to your database
enroll_speaker("Divne", r'C:\Users\HP\Documents\Dash\record.wav')
