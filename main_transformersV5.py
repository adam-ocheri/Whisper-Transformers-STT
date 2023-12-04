import transformers
import datasets
from datasets import load_dataset
import evaluate
import gradio
from pydub import AudioSegment
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(
    device
)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")


print("Started Inference...")

# sample = load_dataset("osanseviero/dummy_ja_audio")["train"]["audio"][0]
# speech_data = sample["array"]
# print("Sample Data:")
# print(sample)
# print("Speech Data:")
# for data in speech_data:
#     print(data)
# speech_data = None


# Provide the path to your audio file
# audio_file_path = "conv-test-00_chunk-0.mp3"
def begin_transcribe(audio_file_path):
    # Load the audio file
    speech_data, sr = librosa.load(audio_file_path)
    # speech_data = AudioSegment.from_mp3(audio_file_path)

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

    print("Result is: ", result)
    print("|-> FINISH <-|")


def batch_transcribe(chunks_name, max_idx):
    for i in range(0, max_idx + 1):
        begin_transcribe(f"{chunks_name}{i}.mp3")


batch_transcribe("conv-test-00_chunk-", 4)
