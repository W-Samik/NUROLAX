# --- START OF FILE backend/main.py ---
import os
import shutil
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import prediction functions from your modules
from dysarthia import predict_dysarthria, dysarthria_model
from keystockes import predict_keystroke_risk, keystroke_model, keystroke_scaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_KEYSTROKES_DIR = Path("temp_keystrokes")
FRONTEND_DIR = Path(r"Ai_models\himachal_model\website\frontend") # Relative path to the frontend directory

# Create temporary directories if they don't exist
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
TEMP_KEYSTROKES_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Parkinson Analysis API")

# --- CORS Middleware ---
# Allow requests from your frontend (adjust origins if needed)
origins = [
    r"http://localhost",        # Common for local dev
    r"http://localhost:8000",   # If frontend served separately
    r"http://127.0.0.1",
    r"http://127.0.0.1:8000",
    # Add the origin where your HTML file will be served from,
    # e.g., file:/// if opened directly, or a specific port if served by another tool
    "null", # Needed for file:/// origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Check if models loaded ---
if dysarthria_model is None:
    logging.warning("Dysarthria model failed to load. Voice analysis endpoint will be disabled.")
if keystroke_model is None or keystroke_scaler is None:
    logging.warning("Keystroke model or scaler failed to load. Keystroke analysis endpoint will be disabled.")


# --- Pydantic Model for Keystroke Data ---
class KeystrokeData(BaseModel):
    csv_data: str

# --- API Endpoints ---

# Serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
         return HTMLResponse(content="<html><body><h1>Error</h1><p>index.html not found.</p></body></html>", status_code=404)
    return FileResponse(index_path)

# Serve static files (CSS, JS)
app.mount(r"/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Endpoint for Voice Analysis
@app.post(r"/analyze/voice")
async def analyze_voice_endpoint(audio_file: UploadFile = File(...)):
    if dysarthria_model is None:
        raise HTTPException(status_code=503, detail="Voice analysis service unavailable (model not loaded).")

    # Generate a unique filename
    file_ext = Path(audio_file.filename).suffix or ".wav" # Default to .wav if no extension
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_file_path = TEMP_AUDIO_DIR / temp_filename

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logging.info(f"Saved temporary audio file: {temp_file_path}")

        # Run prediction
        probability = predict_dysarthria(str(temp_file_path))

        if probability is None:
             raise HTTPException(status_code=500, detail="Voice analysis failed during prediction.")

        # Define a simple threshold (adjust as needed based on model validation)
        threshold = 0.5
        diagnosis = "Possible Dysarthria" if probability >= threshold else "Likely Non-Dysarthria"

        return JSONResponse(content={
            "probability": probability,
            "diagnosis": diagnosis,
            "detail": f"Probability score: {probability:.3f}"
        })

    except Exception as e:
        logging.error(f"Error processing voice analysis request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during voice analysis: {e}")

    finally:
        # Clean up the temporary file
        if temp_file_path.exists():
            try:
                 temp_file_path.unlink()
                 logging.info(f"Deleted temporary audio file: {temp_file_path}")
            except Exception as e_del:
                 logging.error(f"Error deleting temporary audio file {temp_file_path}: {e_del}")
        # Ensure file object is closed (FastAPI usually handles this)
        await audio_file.close()


# Endpoint for Keystroke Analysis
@app.post(r"/analyze/keystrokes")
async def analyze_keystrokes_endpoint(data: KeystrokeData):
    if keystroke_model is None or keystroke_scaler is None:
        raise HTTPException(status_code=503, detail="Keystroke analysis service unavailable (model/scaler not loaded).")

    csv_string = data.csv_data
    if not csv_string or len(csv_string.splitlines()) < 2: # Basic check for some data
         raise HTTPException(status_code=400, detail="Received empty or invalid CSV data.")

    # Optional: Save CSV to temp file for consistency or debugging
    # temp_filename = f"{uuid.uuid4()}.csv"
    # temp_file_path = TEMP_KEYSTROKES_DIR / temp_filename
    # try:
    #     with open(temp_file_path, "w") as f:
    #         f.write(csv_string)
    #     logging.info(f"Saved temporary keystroke file: {temp_file_path}")
    #     # Modify predict_keystroke_risk to take file_path if using this approach
    #     risk_score = predict_keystroke_risk(str(temp_file_path))
    # except ...
    # finally:
    #     if temp_file_path.exists(): temp_file_path.unlink()

    # Direct prediction from string
    try:
        risk_score = predict_keystroke_risk(csv_string)

        if risk_score is None:
            raise HTTPException(status_code=500, detail="Keystroke analysis failed during prediction.")

        # Define a simple threshold (adjust based on model validation)
        threshold = 0.5
        risk_level = "Higher Risk Indicated" if risk_score >= threshold else "Lower Risk Indicated"

        return JSONResponse(content={
            "risk_score": risk_score,
            "risk_level": risk_level,
            "detail": f"Predicted risk score: {risk_score:.3f}"
        })

    except Exception as e:
        logging.error(f"Error processing keystroke analysis request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during keystroke analysis: {e}")

# --- Add health check endpoint (optional) ---
@app.get(r"/health")
async def health_check():
    return {"status": "ok", "voice_model_loaded": dysarthria_model is not None, "keystroke_model_loaded": keystroke_model is not None and keystroke_scaler is not None}

# --- Main execution block (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server...")
    # Use host="0.0.0.0" to make it accessible on your network
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# --- END OF FILE backend/main.py ---