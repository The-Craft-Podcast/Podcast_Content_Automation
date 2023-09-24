import os
import sys
import ast

def load_transcript(file_path: str) -> list:
    """
    Load the transcript from the file.
    """
    segments = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Speaker:" in line:
                segments.append(line.strip())
    return segments

def identify_speakers(segments: list) -> set:
    """
    Identify the speakers in the provided file.
    """
    unique_speakers = set()
    for segment in segments:
        speaker = segment.split(":")[1].strip()
        unique_speakers.add(speaker)
    return unique_speakers

def replace_speakers(transcript_file_path: str, speaker_mapping: dict, new_file_path: str) -> None:
    """
    Replace the speaker labels with the real names.
    """
    # Read the transcript
    with open(transcript_file_path, 'r') as f:
        transcript_content = f.read()

    # Replace each instance of the speaker label with the corresponding real name
    for speaker_label, real_name in speaker_mapping.items():
        transcript_content = transcript_content.replace(f"Speaker: {speaker_label}", f"Speaker: {real_name}")

    # Save the updated transcript to a new file
    with open(new_file_path, 'w') as f:
        f.write(transcript_content)

    print("\nSpeaker Replacement Done!\n")
    return None

def get_user_input_as_dict():
    """
    Ask the user to input a dictionary in the format of {'key1': 'value1', 'key2': 'value2'}.
    If the input is invalid, ask the user to re-enter.
    Return the dictionary.
    """
    try:
        user_input = input("Please enter a speaker mapping (e.g., {'SPEAKER_00': 'John Wick', 'SPEAKER_01': 'Tom Cruise'}): ")
        user_dict = ast.literal_eval(user_input)
        if not isinstance(user_dict, dict):
            raise ValueError
        return user_dict
    except (SyntaxError, ValueError):
        print("Invalid dictionary format. Please try again.")
        return get_user_input_as_dict()

if __name__ == '__main__':
    # Original transcript file path
    transcript_input_path = "../Transcripts/output_William.txt"
    # Output file path
    transcript_output_path = "../Transcripts/output_William_corrected.txt"

    # Load the transcript without correct speaker names
    segments = load_transcript(transcript_input_path)
    # Pick out the unique speaker tags to help you assign speaker mapping
    unique_speakers = identify_speakers(segments)
    print(unique_speakers)

    # Get speaker_mapping from user input in terminal
    speaker_mapping = get_user_input_as_dict()
    ## Example speaker_mapping ##
    # speaker_mapping = {
    #     "SPEAKER_00": "Michael Du",
    #     "SPEAKER_01": "Daniel Tedesco",
    #     "SPEAKER_02": "William Entriken",
    #     "Unknown": "Unknown"
    # }
    
    # Replace the speaker labels with the real names
    replace_speakers(transcript_input_path, speaker_mapping, transcript_output_path)

    ## END ##
