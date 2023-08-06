# Podcast Editing Automation With LLMs

This project explores automating the process of editing podcast interviews with the help of AI models.
Usually, the process of editing a podcast interview content comes down to:
1. Split audio from video
2. Generate accurate transcripts with correct speaker names
3. Analyze the interview (video/audio) and edit it into multiple clips with each has a specific topics or highlights
4. Generate Chapters for each of the clip as well
5. Write an article about the interview to publish on media such as Medium
6. Write out intriguing social media posts for each clip to spread it out in public

This project aims to automate most of the editing work with multiple AI models, including [Whisper](https://github.com/openai/whisper) from OpenAI, [WhisperX](https://github.com/m-bain/whisperX) from Oxford, LLMs such as Claude or GPT3.5 and GPT4 (API), and other useful tools such as LangChain and VectorDB. 
Below is a rough draft of the system design of this project:<br><br>

![Workflow](./SysDesign.png)<br><br>

Environment Setup: 
Suggested Python Version: 3.10 & PyTorch 2.0 to match WhisperX

## Part I: Video Transcription and Speaker Diarization with Whisper/WhisperX

Refer to Audio_Transcript Folder
- You can extract audio from the video with `video_to_audio.py`
- Audio transcription and diarization (speaker assignment) with `video_transcription.py` (Need Whisper and WhisperX properly installed and imported)