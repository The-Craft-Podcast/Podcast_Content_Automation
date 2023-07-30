from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

if __name__ == "__main__":
    extract_audio("/Users/mike24dzy/Downloads/The Craft Podcast Interview with William Entriken.mp4", "/Users/mike24dzy/Downloads/William.wav")
