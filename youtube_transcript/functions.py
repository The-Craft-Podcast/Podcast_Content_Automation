import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(url: str):
    '''Fetch youtube video transcript with youtube transcrip api'''
    match = re.search(r'(watch\?v=)(.*)', url)
    video_id = ''
    error_message = "Invalid Youtube Url"
    if match:
        video_id = match.group(2)
    else:
        return error_message
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except:
        print("There was an error with the YouTube Transcript API call.")
        return
    
    transcript_text = ""
    for seg in transcript:
        transcript_text += 'Timestamp: ' + f"{seg['start']}" + "\n" + seg['text'] + "\n"

    return transcript_text
