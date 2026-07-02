from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
from utils.llm import get_groq_client

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Groq's Whisper API."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check if it's an audio file
    if not file.content_type or not file.content_type.startswith("audio/"):
        # Browsers might send webm or ogg which are valid
        if not any(file.filename.endswith(ext) for ext in [".webm", ".ogg", ".mp3", ".wav", ".m4a"]):
            raise HTTPException(status_code=400, detail="Must be an audio file.")

    client = get_groq_client()
    
    try:
        # Create a temporary file to hold the audio
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe with Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(file.filename, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en",
            )
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return {"text": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
