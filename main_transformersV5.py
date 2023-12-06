import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

cache_dir = "./Model/"
model_name = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(
    model_name, cache_dir=cache_dir + model_name + "model"
).to(device)

processor = WhisperProcessor.from_pretrained(
    model_name, cache_dir=cache_dir + model_name + "processor"
)


print("Started Inference...")

global text
text = ""


def begin_transcribe(audio_file_path):
    # Load the audio file
    speech_data, sr = librosa.load(audio_file_path)

    inputs = processor.feature_extractor(
        speech_data, return_tensors="pt", sampling_rate=16_000
    ).input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="he", task="transcribe"
    )

    predicted_ids = model.generate(
        inputs, max_length=480_000, forced_decoder_ids=forced_decoder_ids
    )
    result = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True, normalize=True
    )[0]

    global text
    text += result + "\n"
    print("Result is: ", result[::-1])
    print("|-> FINISH <-|")


def batch_transcribe(chunks_name, max_idx):
    for i in range(0, max_idx + 1):
        begin_transcribe(f"{chunks_name}{i}.mp3")


batch_transcribe("c0_service-person_otr4_chunk-", 11)

with open("TranscriptionResult.txt", "w", encoding="utf-8") as file:
    file.write(text)
