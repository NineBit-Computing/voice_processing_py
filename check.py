# #librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.


#shows audio frequency

# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# # Load the audio file
# audio_path = 'Recording2.wav'  # Replace 'your_audio_file.wav' with your actual audio file path
# y, sr = librosa.load(audio_path)

# # Calculate the Short-Time Fourier Transform (STFT) of the audio signal
# D = librosa.stft(y)

# # Convert the amplitude spectrogram to dB scale
# DB = librosa.amplitude_to_db(abs(D))

# # Plot the spectrogram
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.tight_layout()
# plt.show()

#show audio waveform

# import librosa
# import matplotlib.pyplot as plt

# # Load the audio file
# audio_path = 'Recording2.wav'  # Replace 'your_audio_file.wav' with your actual audio file path
# y, sr = librosa.load(audio_path)

# # Create a time axis in seconds
# time = librosa.times_like(y, sr=sr)

# # Plot the waveform
# plt.figure(figsize=(10, 4))
# plt.plot(time, y, color='b')
# plt.title('Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.xlim(time[0], time[-1])
# plt.tight_layout()
# plt.show()

# from pydub import AudioSegment
# import matplotlib.pyplot as plt

# # Load the audio file
# audio_path = 'Recording2.wav'  
# audio = AudioSegment.from_wav(audio_path)

# # Slow down the audio by 50%
# slower_audio = audio.speedup(playback_speed=0.5)

# # Export the slowed down audio to a temporary WAV file
# temp_path = 'slowed_down_audio.wav'
# slower_audio.export(temp_path, format='wav')

# # Load the slowed down audio using librosa just for visualization
# import librosa

# y, sr = librosa.load(temp_path)
# time = librosa.times_like(y, sr=sr)

# # Plot the waveform
# plt.figure(figsize=(10, 4))
# plt.plot(time, y, color='b')
# plt.title('Waveform after Slowing Down')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.xlim(time[0], time[-1])
# plt.tight_layout()
# plt.show()

##############################################################################

import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Load the audio file
audio_path = 'Recording2.wav'  
audio = AudioSegment.from_wav(audio_path)

y, sr = librosa.load(audio_path)
time1 = librosa.times_like(y, sr=sr)
# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time1, y, color='b')
plt.title('Normalized Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time1[0], time1[-1])
plt.tight_layout()
plt.show()

# Normalize the audio to increase the volume
normalized_audio = audio.apply_gain(+15.0)  # Adjust gain as needed

# Export the normalized audio to a temporary WAV file
temp_path = 'clear_audio.wav'
# audio1= AudioSegment.from_wav(temp_path)
# slower_audio = audio1.speed(factor=0.5)
normalized_audio.export(temp_path, format='wav')


# Load the normalized audio
y, sr = librosa.load(temp_path)

# Create a time axis in seconds
time = librosa.times_like(y, sr=sr)

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, y, color='b')
plt.title('Normalized Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time[0], time[-1])
plt.tight_layout()
plt.show()


