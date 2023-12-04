from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa

model_cache_dir = "./Model/model"
token_cache_dir = "./Model/tokenizer"

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3", cache_dir=model_cache_dir
)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", cache_dir=token_cache_dir
).to("cuda")

filename = "conv-test-00.mp3"  # replace with your file name
audio, rate = librosa.load(filename, sr=None)

input_features = processor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features

with torch.no_grad():
    predicted_ids = model.generate(inputs=input_features.to("cuda"))[0]

transcription = processor.decode(predicted_ids)

print("FINISHED")
print(transcription)
