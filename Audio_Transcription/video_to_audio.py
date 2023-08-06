from moviepy.editor import VideoFileClip

# Using this script to first extract the audio.wav from the video you wish to transcribe if you have a video on your local machine
# Then run video_transcription.py to load your audio.wav and also define a new transcript output path to save the results

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

if __name__ == "__main__":
    extract_audio("/Users/mike24dzy/Downloads/The Craft Podcast Interview with William Entriken.mp4", "/Users/mike24dzy/Downloads/William.wav")
