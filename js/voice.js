// Voice Recording Functionality
let mediaRecorder;
let audioChunks = [];

document.getElementById('recordButton').addEventListener('click', async () => {
  const button = document.getElementById('recordButton');
  const status = document.getElementById('recordingStatus');
  const audioPlayback = document.getElementById('audioPlayback');
  
  if (!mediaRecorder) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioPlayback.src = URL.createObjectURL(audioBlob);
        audioPlayback.style.display = 'block';
        
        // Simulate analysis (replace with actual API call)
        simulateVoiceAnalysis();
      };
      
      mediaRecorder.start();
      button.textContent = 'Stop Recording';
      button.classList.add('recording');
      status.style.display = 'block';
      
      // Auto-stop after 5 seconds for analysis
      setTimeout(() => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          stream.getTracks().forEach(track => track.stop());
        }
      }, 5000);
      
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Microphone access denied. Please enable permissions.");
    }
  } else {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    mediaRecorder = null;
    audioChunks = [];
    button.textContent = 'Start Recording';
    button.classList.remove('recording');
    status.style.display = 'none';
  }
});

// Simulated Analysis Function (Replace with real API call)
function simulateVoiceAnalysis() {
  const resultDiv = document.getElementById('analysisResult');
  const tremorElement = document.getElementById('tremorFreq');
  const amplitudeElement = document.getElementById('amplitudeVar');
  
  // Simulate processing delay
  setTimeout(() => {
    // These would come from your actual analysis API
    const tremorFreq = (Math.random() * 5 + 3).toFixed(1);
    const amplitudeVar = (Math.random() * 30 + 5).toFixed(1);
    
    tremorElement.textContent = `${tremorFreq} Hz`;
    amplitudeElement.textContent = `${amplitudeVar}%`;
    
    resultDiv.style.display = 'block';
    
    // Highlight potential issues
    if (tremorFreq > 6.5 || amplitudeVar > 25) {
      resultDiv.style.borderLeft = "4px solid #dc3545";
    } else {
      resultDiv.style.borderLeft = "4px solid #28a745";
    }
  }, 1500);
}
