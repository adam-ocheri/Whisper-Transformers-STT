from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load

librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

model_cache_dir = "./Model/model"
token_cache_dir = "./Model/tokenizer"

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3", cache_dir=model_cache_dir
)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", cache_dir=token_cache_dir
).to("cuda")


def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["reference"] = processor.tokenizer._normalize(batch["text"])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch


result = librispeech_test_clean.map(map_to_pred)

print(predictions=result["prediction"])


# wer = load("wer")
# print(
#     100 * wer.compute(references=result["reference"], predictions=result["prediction"])
# )
