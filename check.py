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

# import matplotlib.pyplot as plt
# from pydub import AudioSegment
# import speech_recognition as ss
# import noisereduce as nr
# import librosa

# # Load the audio file
# audio_path = 'voice.wav'
# audio = AudioSegment.from_wav(audio_path)

# # Plot the waveform before normalization
# plt.figure(figsize=(10, 4))
# plt.plot(audio.get_array_of_samples(), color='b')
# plt.title('Waveform before Normalization')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()

# # Apply noise reduction
# audio_data = audio.get_array_of_samples()
# reduced_noise = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)

# # Export the noise-reduced audio to a temporary WAV file
# temp_path = 'clear_audio.wav'
# normalized_audio = AudioSegment(
#     reduced_noise.tobytes(),
#     frame_rate=audio.frame_rate,
#     sample_width=audio.sample_width,
#     channels=audio.channels
# )
# # normalized_audio = normalized_audio + 16
# normalized_audio.export(temp_path, format='wav')

# # Load the normalized audio using librosa
# y, sr = librosa.load(temp_path)

# # Plot the waveform after normalization
# plt.figure(figsize=(10, 4))
# plt.plot(y, color='b')
# plt.title('Waveform after Normalization')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()

# # Perform speech-to-text conversion using the normalized audio
# r = ss.Recognizer()
# with ss.AudioFile(temp_path) as source:
#     audio_data = r.record(source)  # Load audio to memory
#     text = r.recognize_google(audio_data)

# print("Text from speech:", text)


import matplotlib.pyplot as plt
from pydub import AudioSegment
import speech_recognition as ss
from gramformer import Gramformer
import librosa
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(5212)

def correct_text_with_gramformer(text):
    # Initialize Gramformer
    gramformer = Gramformer(models=1)

    # Split the text into smaller chunks
    chunk_size = 200  # Adjust the chunk size as needed
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Correct each chunk and concatenate the results
    corrected_chunks = []
    for chunk in chunks:
        corrected_sentences = gramformer.correct(chunk)
        corrected_sentences_str = ' '.join(corrected_sentences)
        corrected_chunks.append(corrected_sentences_str)

    corrected_text = ' '.join(corrected_chunks)

    return corrected_text

# Load the audio file
audio_path = 'voice.wav'
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
normalized_audio.export(temp_path, format='wav')

# Load the normalized audio
y, sr = librosa.load(temp_path)
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

# Perform speech-to-text conversion using the normalized audio
r = ss.Recognizer()
with ss.AudioFile(temp_path) as source:
    audio_data = r.record(source)  # Load audio to memory
    text = r.recognize_google(audio_data)

print("Text from speech:", text)

# Correct grammatical errors in the recognized text
corrected_text = correct_text_with_gramformer(text)
print("Corrected text:", corrected_text)



