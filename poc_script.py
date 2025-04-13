import requests
import os
import logging
import wave
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8000"  # Assuming FastAPI runs on default port 8000 locally
VOICE_ENDPOINT = f"{BASE_URL}/analyze/voice"
KEYSTROKE_ENDPOINT = f"{BASE_URL}/analyze/keystrokes"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

# --- Sample Data ---

# 1. Keystroke Data
sample_keystroke_csv_string = """
"Shift",0.100,463.100
"A",136.200,246.300
" ",808.600,864.900
"m",1034.700,1093.400
"e",1183.800,1264.800
"a",1386.200,1440.100
"n",1717.200,1763.200
"w",2363.000,2437.800
"h",2643.100,2818.800
"i",2809.300,2889.200
"u",2820.000,2884.500
"Backspace",3385.000,3491.100
"l",4626.500,4679.700
"e",4718.600,4766.500
" ",4800.400,4869.200
"p",5376.100,5449.500
"r",5479.500,5576.500
"e",5576.900,5658.000
"s",5847.100,5936.700
"e",6141.600,6257.600
"d",6726.100,6824.400
"e",6889.400,6975.600
"n",7119.700,7179.000
"t",7298.300,7378.300
" ",7425.700,7487.500
"d",7687.100,7754.700
"i",8043.300,8116.300
"s",8129.400,8220.100
"c",8400.100,8472.400
"o",8583.600,8664.100
"v",9078.900,9129.500
"e",9203.500,9315.300
r",9347.000,9449.500
"e",9501.200,9600.000
"d",9707.600,9802.700
" ",9887.700,9984.600
"q",10196.800,10307.100
"u",10597.100,10651.900
"i",10815.600,10863.600
"e",11009.700,11108.200
"t",11295.200,11369.600
"l",11910.600,11961.100
"y",12009.400,12098.600
"""

# 2. Audio Data (Generate a dummy WAV file or specify path to a real one)
DUMMY_AUDIO_PATH = "dummy_audio.wav"

def create_dummy_wav(filename, duration=3, sample_rate=16000):
    """Creates a simple sine wave WAV file."""
    if os.path.exists(filename):
        logging.info(f"Dummy audio file '{filename}' already exists.")
        return
    try:
        logging.info(f"Creating dummy audio file: {filename}")
        frequency = 440  # A4 note
        n_samples = duration * sample_rate
        t = np.linspace(0., duration, n_samples)
        amplitude = np.iinfo(np.int16).max * 0.5 # Use half max amplitude
        data = amplitude * np.sin(2. * np.pi * frequency * t)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(data.astype(np.int16).tobytes())
        logging.info(f"Dummy audio file created successfully.")
    except Exception as e:
        logging.error(f"Failed to create dummy WAV file: {e}", exc_info=True)

create_dummy_wav(DUMMY_AUDIO_PATH)

# --- Test Functions ---

def test_health_check():
    """Tests the health check endpoint."""
    logging.info(f"\n--- Testing Health Check [{HEALTH_ENDPOINT}] ---")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Health Check Request Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during health check: {e}")

def test_keystroke_analysis():
    """Sends sample keystroke data to the analysis endpoint."""
    logging.info(f"\n--- Testing Keystroke Analysis [{KEYSTROKE_ENDPOINT}] ---")
    payload = {"csv_data": sample_keystroke_csv_string}
    try:
        response = requests.post(KEYSTROKE_ENDPOINT, json=payload, timeout=30) # Longer timeout for prediction
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Keystroke Analysis Request Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during keystroke analysis: {e}")

def test_voice_analysis(audio_file_path):
    """Sends a sample audio file to the voice analysis endpoint."""
    logging.info(f"\n--- Testing Voice Analysis [{VOICE_ENDPOINT}] ---")
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}. Skipping voice analysis test.")
        return

    files = {'audio_file': (os.path.basename(audio_file_path), open(audio_file_path, 'rb'), 'audio/wav')}
    try:
        # Note: 'files' argument is used for multipart/form-data (file uploads)
        response = requests.post(VOICE_ENDPOINT, files=files, timeout=60) # Even longer timeout for audio processing
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Voice Analysis Request Failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during voice analysis: {e}")
    finally:
        # Ensure the file handle opened in 'files' is closed
        if 'audio_file' in files:
             files['audio_file'][1].close()


# --- Run Tests ---
if __name__ == "__main__":
    print("=== Running POC Client Tests ===")
    print(f"Targeting backend API at: {BASE_URL}")

    test_health_check()
    test_keystroke_analysis()
    test_voice_analysis(DUMMY_AUDIO_PATH) # Use the dummy audio file

    print("\n=== POC Client Tests Finished ===")
