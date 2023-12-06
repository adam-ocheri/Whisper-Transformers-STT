from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    """Normalize given audio chunk"""
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split_audio_file(
    filepath,
    min_silence_len=2000,
    silence_thresh=-16,
    silence_duration=500,
    target_dBFS=-20,
):
    # Load your audio.
    audio_file = AudioSegment.from_mp3(f"{filepath}.mp3")

    # Split track where the silence is 2 seconds or more and get chunks using
    # the imported function.
    chunks = split_on_silence(
        # Use the loaded audio.
        audio_file,
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long (default value)
        min_silence_len=min_silence_len,
        # Consider a chunk silent if it's quieter than -16 dBFS (default value)
        # (You may want to adjust this parameter.)
        silence_thresh=silence_thresh,
    )

    cumulative_time = 0  # to keep track of the cumulative time

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=silence_duration)
        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, target_dBFS)

        # TODO: Extract precise timestamp of this chunk, in relation to the base source file
        time = cumulative_time
        cumulative_time += len(chunk)
        # os.mkdir(filepath)
        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.mp3.".format(i), f"Start Time: {time * 0.001}")
        normalized_chunk.export(
            f".//{filepath}_chunk-{i}.mp3", bitrate="192k", format="mp3"
        )


# This is the "ok values" for now
split_audio_file("c0_service-person_otr4", 900, -40, 1200, -20)

# Here I test! will you send me an angel?
# split_audio_file("c0_service-person_otr4S", 300, -35, 0, -20)
