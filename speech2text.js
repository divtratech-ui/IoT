let mediaRecorder;
let audioChunks = [];
const voiceBtn = document.getElementById("voiceBtn");
const speechOutput = document.getElementById("speechOutput");

voiceBtn.addEventListener("mousedown", async () => {
    audioChunks = [];
    speechOutput.textContent = "Listening... Release button to send.";
    voiceBtn.style.background = "#ff4d4d";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            speechOutput.textContent = "Processing voice patterns...";
            await sendAudioToServer(audioBlob);
        };

        mediaRecorder.start();
    } catch (err) {
        console.error("Microphone access denied:", err);
        speechOutput.textContent = "Microphone access denied.";
    }
});

voiceBtn.addEventListener("mouseup", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        voiceBtn.style.background = ""; // reset button style
    }
});

async function sendAudioToServer(blob) {
    const formData = new FormData();
    formData.append("file", blob, "voice_input.wav");

    try {
        const response = await fetch("http://localhost:8000/verify", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.verified) {
            const command = data.text.toLowerCase().trim();
            speechOutput.textContent = `Hello ${data.user}! You said: "${command}"`;

            // Process commands
            if (command.includes("on") && command.includes("light")) {
                client.publish(controlTopic, "ON");
            } else if (command.includes("off") && command.includes("light")) {
                client.publish(controlTopic, "OFF");
            } else {
                alert(`Command not recognized: "${command}"`);
            }
        } else {
            speechOutput.textContent = "Access Denied: Voice biometric verification failed.";
            alert("Unauthorized User! Match score was too low.");
        }
    } catch (error) {
        console.error("Error connecting to verification server:", error);
        speechOutput.textContent = "Error: Backend server offline.";
    }
}