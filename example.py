# import torchaudio
# import matplotlib.pyplot as plt
# from audio_denoiser.AudioDenoiser import AudioDenoiser

# # Load the noisy waveform
# noisy_waveform, sample_rate = torchaudio.load('Recording2.wav')

# # Instantiate the denoiser
# denoiser = AudioDenoiser()

# # Process the waveform
# denoised_waveform = denoiser.process_waveform(noisy_waveform, sample_rate, auto_scale=False)

# # Plot the noisy waveform
# plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)
# plt.plot(noisy_waveform.t().numpy())

# plt.title('Noisy Waveform')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# # Plot the denoised waveform
# plt.subplot(2, 1, 2)
# plt.plot(denoised_waveform.t().numpy())

# plt.title('Denoised Waveform')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()


# import language_tool_python

# mytext = """
# I is testng grammar tool using python. It does not costt anythng.
# """

# def grammarCorrector(text):
#     tool = language_tool_python.LanguageTool('en-US')
#     result = tool.correct(text)
#     return result

# output_data = grammarCorrector(mytext)
# print(output_data)


# from gramformer import Gramformer
# import torch

# def set_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(5212)

# def correct_text_with_gramformer(text):
#     # Initialize Gramformer
#     gramformer = Gramformer(models=1)

#     # Split the text into smaller chunks
#     chunk_size = 200  # Adjust the chunk size as needed
#     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

#     # Correct each chunk and concatenate the results
#     corrected_chunks = []
#     for chunk in chunks:
#         corrected_sentences = gramformer.correct(chunk)
#         corrected_sentences_str = ' '.join(corrected_sentences)
#         corrected_chunks.append(corrected_sentences_str)

#     corrected_text = ' '.join(corrected_chunks)

#     return corrected_text

# text = "having a best friend by Sonu Sudheer is one of the best Friendship moral stories in  English the stories about two friends who were Walking Through the desert during the journey they argued over something and one friends left the other the one who got slap was heard by testes of his best friend but did not react quickly brought in the send today my best friend slap me after some time they found an Oscars and they started taking a bath in the lakes suddenly the one who had been slapped started drowning then his friends come to his rescue and saved him after he recovered from the drowning he encrypt today my best friend save my life on a stone"

# corrected_text = correct_text_with_gramformer(text)
# print("Corrected text:", corrected_text)

############################## speech to text - torch#######################################################
# import torch
# # import zipfile
# # import torchaudio
# from glob import glob

# device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
# model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                        model='silero_stt',
#                                        language='en', # also available 'de', 'es'
#                                        device=device)
# (read_batch, split_into_batches,
#  read_audio, prepare_model_input) = utils  # see function signature for details

# # download a single file in any format compatible with TorchAudio
# # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
# #                                dst ='speech_orig.wav', progress=True)
# test_files = glob('voice.wav')
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]),
#                             device=device)

# output = model(input)
# for example in output:
#     print(decoder(example.cpu()))


############################speech to text - tensorflow #####################################################


# import os
# import torch
# import subprocess
# import tensorflow as tf
# import tensorflow_hub as tf_hub
# from omegaconf import OmegaConf

# language = 'en' # also available 'de', 'es'

# # load provided utils using torch.hub for brevity
# _, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language=language)
# (read_batch, split_into_batches,
#  read_audio, prepare_model_input) = utils

# # see available models
# torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
# models = OmegaConf.load('models.yml')
# available_languages = list(models.stt_models.keys())
# assert language in available_languages

# # load the actual tf model
# torch.hub.download_url_to_file(models.stt_models.en.latest.tf, 'dev-clean-tar.gz')
# subprocess.run('rm -rf tf_model && mkdir tf_model && tar xzfv tf_model.tar.gz -C tf_model',  shell=True, check=True)
# tf_model = tf.saved_model.load('tf_model')

# # download a single file in any format compatible with TorchAudio
# # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav', dst ='speech_orig.wav', progress=True)
# test_files = ['voice3.wav']
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]))

# # tf inference
# res = tf_model.signatures["serving_default"](tf.constant(input.numpy()))['output_0']
# print(decoder(torch.Tensor(res.numpy())[0]))

##################################################################################################

# import whisper

# model = whisper.load_model("base")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("voice3.wav")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)

# importing libraries 
# import speech_recognition as sr 
# import os 
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import torch
# from glob import glob

# # create a speech recognition object
# r = sr.Recognizer()

# # a function to recognize speech in the audio file
# # so that we don't repeat ourselves in in other functions
# def transcribe_audio(path):
#     # use the audio file as the audio source
#     with sr.AudioFile(path) as source:
#         audio_listened = r.record(source)
#         # try converting it to text
#         text = r.recognize_google(audio_listened)
#     return text

# # a function that splits the audio file into chunks on silence
# # and applies speech recognition
# def get_large_audio_transcription_on_silence(path):
#     """Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks"""
#     # open the audio file using pydub
#     sound = AudioSegment.from_file(path)  
#     # split audio sound where silence is 500 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#         # experiment with this value for your target audio file
#         min_silence_len = 500,
#         # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
#         # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk 
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         # recognize the chunk
#         try:
#             text = transcribe_audio(chunk_filename)
#         except sr.UnknownValueError as e:
#             print("Error:", str(e))
#         else:
#             text = f"{text.capitalize()}. "
#             # print(chunk_filename, ":", text)
#             whole_text += text
#     # return the text for all chunks detected
#     return whole_text

# path = "voice3.wav"
# print("\nFull text:", get_large_audio_transcription_on_silence(path))


# # Perform speech-to-text conversion using the pytorch library
# device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
# model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                        model='silero_stt',
#                                        language='en', # also available 'de', 'es'
#                                        device=device)
# (read_batch, split_into_batches,
#  read_audio, prepare_model_input) = utils  # see function signature for details

# test_files = glob('voice3.wav')
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]),
#                             device=device)

# output = model(input)
# for example in output:
#     print(decoder(example.cpu()))

# from faster_whisper import WhisperModel

# model_size = "large-v3"

# # Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# # or run on GPU with INT8
# # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
# # model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("voice1.mp3", beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

####################Model - whisper##########

import whisper

# model = whisper.load_model("base")
# result = model.transcribe("samyak1.mp4")
# print(result["text"])

####################Model - faster whisper##########

# from faster_whisper import WhisperModel

# model = WhisperModel("distil-medium.en")

# segments, info = model.transcribe("samyak1.mp4")
# for segment in segments:
#     # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     print(segment.text)


# ############################## speech to text - torch#######################################################
# import torch
# from glob import glob

# device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
# model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                        model='silero_stt',
#                                        language='en', # also available 'de', 'es'
#                                        device=device)
# (read_batch, split_into_batches,
#  read_audio, prepare_model_input) = utils 

# test_files = glob('samyak1.mp4')
# batches = split_into_batches(test_files, batch_size=10)
# input = prepare_model_input(read_batch(batches[0]),
#                             device=device)

# output = model(input)
# for example in output:
#     print(decoder(example.cpu()))