from datasets import load_dataset, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
import librosa

# librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

model_cache_dir = "./Model/model"
token_cache_dir = "./Model/tokenizer"
audio_files = [{"audio": "conv-test-00.mp3"}]
ds = Dataset.from_list(audio_files)
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

# def map_to_pred(batch):
#     audio = batch["audio"]
#     input_features = processor(
#         audio, sampling_rate=16000, return_tensors="pt"
#     ).input_features
#     batch["reference"] = processor.tokenizer._normalize(batch["text"])

#     with torch.no_grad():
#         predicted_ids = model.generate(input_features.to("cuda"))[0]
#     transcription = processor.decode(predicted_ids)
#     batch["prediction"] = processor.tokenizer._normalize(transcription)
#     return batch


# result = ds.map(map_to_pred)

# print(predictions=result["prediction"])


# wer = load("wer")
# print(
#     100 * wer.compute(references=result["reference"], predictions=result["prediction"])
# )
