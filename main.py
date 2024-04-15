##################### Mic using Wisper #################################################

# import pyaudio
# import wave
# import whisper

# print("Whisper Output")

# # Function to record audio from microphone
# def record_audio(filename, seconds=5, chunk=1024, channels=1, rate=44100):
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=channels,
#                     rate=rate,
#                     input=True,
#                     frames_per_buffer=chunk)
#     print("Recording...")
#     frames = []
#     for i in range(0, int(rate / chunk * seconds)):
#         data = stream.read(chunk)
#         frames.append(data)
#     print("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#     wf.setframerate(rate)
#     wf.writeframes(b''.join(frames))
#     wf.close()

# # Record audio and save it to a file
# record_audio("Recording.wav")

# # Transcribe the recorded audio
# model = whisper.load_model("base")
# result = model.transcribe("Recording.wav")
# print("Transcription:", result["text"])


####################Model - faster whisper################################################

# print("Faster Whisper Output")
# from faster_whisper import WhisperModel

# model = WhisperModel("distil-medium.en")

# segments, info = model.transcribe("Speech_S/samyak13.mp4")
# for segment in segments:
#     # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     print(segment.text)


############################## speech to text - torch######################################

# print("Torch Output")
# import torch
# from glob import glob

# device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
# model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                        model='silero_stt',
#                                        language='en', # also available 'de', 'es'
#                                        device=device)
# (read_batch, split_into_batches,
#  read_audio, prepare_model_input) = utils 

# test_files = glob('Speech_S/samyak13.mp4')
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]),
#                             device=device)

# output = model(input)
# for example in output:
#     print(decoder(example.cpu()))

####################Model - whisper###########################################################

print("Whisper Output")
import whisper

model = whisper.load_model("small")
result = model.transcribe("Speech_B/bharat11.mp4")
print(result["text"])