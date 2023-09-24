# Explore how to transcribe video's audio into a text (with spealer name tag + timestamps + and conversation logs)
import os
import subprocess
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
from pytube import YouTube
import whisper
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
import matplotlib.pyplot as plt
from datetime import datetime
import json

# rewrite into a class
# Download Video Through YouTube
def download_youtube_video(url: str, output_path: Optional[str] = None) -> None:
    yt = YouTube(url)

    video_stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

    if output_path:
        video_stream.download(output_path)
        print(f"Video successfully downloaded to {output_path}")
    else:
        video_stream.download()
        print("Video downloaded to the current directory")

# Convert audio file to WAV format
def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    command = f'ffmpeg -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{output_file}"'

    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully converted "{input_file}" to "{output_file}"')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}, could not convert "{input_file}" to "{output_file}"')


# Transcribe the audio with whisper model
# Returns: A dictionary representing the transcript, including the segments, the language code, and the duration of the audio file.
def transcribe(audio_file: str, model_name: str, device: str = "cpu") -> Dict[str, Any]:

    model = whisper.load_model(model_name, device)
    result = model.transcribe(audio_file)

    language_code = result["language"]
    return {
        "segments": result["segments"],
        "language_code": language_code,
    }

# Align segments
# Returns: A dictionary representing the aligned transcript segments.
def align_segments(segments: List[Dict[str, Any]], language_code: str, audio_file: str, device: str = "cpu") -> Dict[str, Any]:

    model_a, metadata = load_align_model(language_code=language_code, device=device)
    result_aligned = align(segments, model_a, metadata, audio_file, device)
    # print("result_aligned", result_aligned)
    return result_aligned

# Speaker diarization of the audio file
# Returns: A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
def diarize(audio_file: str, hf_token: str) -> Dict[str, Any]:

    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
    diarization_result = diarization_pipeline(audio_file)
    return diarization_result

# Assign speakers to each transcript segment based on the speaker diarization result
# Returns: A list of dictionaries representing each segment of the transcript, including the start and end times, the spoken text, and the speaker ID.
def assign_speakers(diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]) -> List[Dict[str, Any]]:
    # result_segments, word_seg = assign_word_speakers(
    #     diarization_result, aligned_segments["segments"]
    # )
    
    result_segments = assign_word_speakers(diarization_result, aligned_segments)
    # print("result_segments", result_segments)
    results_segments_w_speakers: List[Dict[str, Any]] = []

    segments = result_segments["segments"]
    # print("segments from result_segments:\n", segments)
    for i in segments:
        # print("\nprint i", i)
        speaker = i["speaker"] if "speaker" in i else "Unknown"
        results_segments_w_speakers.append(
            {
                "start": i["start"],
                "end": i["end"],
                "text": i["text"],
                "speaker": speaker,
            }
        )
    # for result_segment in result_segments["segments"]:
    #     results_segments_w_speakers.append(
    #         {
    #             "start": result_segment["start"],
    #             "end": result_segment["end"],
    #             "text": result_segment["text"],
    #             "speaker": result_segment["speaker"],
    #         }
    #     )
    return results_segments_w_speakers

# Putting everything together, transcribe and diarize
# Returns: A list of dictionaries representing each segment of the transcript, including the start and end times, the spoken text, and the speaker ID.
def whisperx_align_and_diarize(audio_file: str, transcript: Dict[str, Any], hf_token: str, output_file: str, device: str = "cpu") -> List[Dict[str, Any]]:

    aligned_segments = align_segments(transcript["segments"], transcript["language_code"], audio_file, device)
    # print("aligned_segments", aligned_segments)
    print("Aligned segments!")

    diarization_result = diarize(audio_file, hf_token)
    # print("Diarization is done!\n", diarization_result)
    print("Diarization is done!")

    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)
    print("Speaker assignment is done!")

    # Instead of printing out the resutls, let's save the result in a text file
    # with open("/Users/mike24dzy/Downloads/test.txt", "w") as f:
    #     for i, segment in enumerate(results_segments_w_speakers):
    #         f.write(f"Segment {i + 1}:\n")
    #         f.write(f"Start time: {segment['start']:.2f}\n")
    #         f.write(f"End time: {segment['end']:.2f}\n")
    #         f.write(f"Speaker: {segment['speaker']}\n")
    #         f.write(f"Transcript: {segment['text']}\n")
    #         f.write("\n")
    # print("The result is saved into Download folder!")
    with open(output_file, "w") as f:
        for i, segment in enumerate(results_segments_w_speakers):
            f.write(f"Segment {i + 1}:\n")
            f.write(f"Start time: {segment['start']:.2f}\n")
            f.write(f"End time: {segment['end']:.2f}\n")
            f.write(f"Speaker: {segment['speaker']}\n")
            f.write(f"Transcript: {segment['text']}\n")
            f.write("\n")
    print("The result is saved into Download folder!")

    return results_segments_w_speakers



# Execution of audio transcribing
if __name__ == "__main__":
    
    # model_names = ["base", "medium", "large"]
    # devices = ["cpu", "cuda"]
    print("Aduio Transcription In Progress")
    ## Set the huggingface API token in current session with 'export HUGGINGFACE_TOKEN=your_token' in the terminal
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    language_code = "en"

    # audio_file = "resistance_converted.wav"
    audio_file = "/Users/mike24dzy/Downloads/William.wav"
    output_file = "../Transcripts/output_William.txt"
    ## Remember to convert the audio file if ran into pyannote error: "ffmpeg -i input.wav -c:a pcm_s16le input_converted.wav"

    # Run the transcription and diarization
    start_time = time.time()
    start_time_datetime = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    print("Start at: ", start_time_datetime)

    # Transcribe audio
    transcript = transcribe(audio_file, 'large', "cpu")
    print("Transcription is done!\n")

    ## Temperarily save the transcription result into json so don't need to run it every time
    with open('transcription_temp.json', 'w') as f:
        json.dump(transcript, f)
    print("Transcript saved to json!\n")
    
    ## Load transcript back in for debugging
    # with open('transcription_temp.json', 'r') as f:
    #     transcript = json.load(f)

    # Align transcript, diarize, assign speakers, and save the result to a text file
    results_segments_w_speakers = whisperx_align_and_diarize(audio_file, transcript, hf_token, output_file, "cpu")
    
    end_time = time.time()
    end_time_datetime = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    print("End at: ", end_time_datetime)

    memory_usage = psutil.Process().memory_info().rss / (1024 ** 3)
    print("Memory Usage: ", memory_usage)
    