import datetime
import os
import wave

import cv2
import numpy as np
import pyaudio

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Open the default camera

# Configure PyAudio for audio capture
audio = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # One audio channel
RATE = 44100  # Sampling rate
CHUNK = 1024  # Buffer size

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

video_frames = []
audio_frames = []

try:
    while True:
        ret, frame = video_capture.read()  # Capture video frame
        cv2.imshow('Video', frame)  # Display video in window

        video_frames.append(frame)  # Store video frame

        # Capture audio
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_frames.append(audio_array.tobytes())

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

print("\nSaving video and audio...")

# Create a directory to store the data if it doesn't exist
if not os.path.exists("captures"):
    os.makedirs("captures")

# Get the current date and time for the file name
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Save the video as an AVI file
video_writer = cv2.VideoWriter(f"captures/video_{timestamp}.avi",
                               cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

for frame in video_frames:
    video_writer.write(frame)

video_writer.release()

# Save the audio as a WAV file
audio_writer = wave.open(f"captures/audio_{timestamp}.wav", "wb")
audio_writer.setnchannels(CHANNELS)
audio_writer.setsampwidth(audio.get_sample_size(FORMAT))
audio_writer.setframerate(RATE)
audio_writer.writeframes(b''.join(audio_frames))
audio_writer.close()

# Release PyAudio resources
stream.stop_stream()
stream.close()
audio.terminate()

print("\nVideo and audio saved successfully!")
