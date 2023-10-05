import os
import sys
from youtube_transcript.functions import get_youtube_transcript
from llm_direct.llm_direct import LLM_Summarization
import tiktoken
import openai
import json
import yaml

openai.api_key = os.environ.get("OPENAI_API_KEY")
url = "https://www.youtube.com/watch?v=XM6NHNxS39E"
yt_transcript = get_youtube_transcript(url)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

token_count = len(encoding.encode(yt_transcript))
print(f"The text contains {token_count} tokens.")


# Test LLM direct calls module
prompt_book_path = "prompts.yml"
# with open(prompt_book_path, 'r') as file:
#     prompt_book = yaml.safe_load(file)

# llm_call = LLM_Summarization(user_input_string=yt_transcript, prompt_book_path=prompt_book_path)

# test_article = llm_call.generate_blog_article()

# test_topic_sum, test_topics = llm_call.generate_topics()
# print(test_topic_sum)
# print(test_topics)
topics = '''
topic-name : Running Startups
description : The speaker has been running startups for 25 years and believes that it is more effective to find people who already like the product or approach and get them excited, rather than trying to change someone's mind from no to yes.

topic-name : Background and Early Years
description : The speaker, Phil Libin, is the co-founder and CEO of All Turtles and Mmhmm. He has been a startup founder for the fifth time and was previously known for starting Evernote. He shares his background as a refugee from the Soviet Union and his early passion for computers and programming.

topic-name : Starting Companies
description : Phil Libin shares his journey of starting companies, including Engine Five, Core Street, and Evernote. He discusses the challenges and lessons learned from each venture, such as the importance of building a product and having a strong point of view.

topic-name : Evernote
description : Phil Libin talks about the creation of Evernote, a productivity app that aimed to help people remember things and be more productive. He highlights the unique features of Evernote, such as background synchronization and searching inside images, that set it apart from other note-taking apps.

topic-name : Building for Yourself
description : Phil Libin recommends that for a first-time startup, it is best to build for yourself or a community that you already understand well. He emphasizes the importance of being the world's biggest expert on the problems you are trying to solve and understanding the needs of your target audience.

topic-name : Changing Behavior
description : Phil Libin believes that successful products or companies require changing people's behavior in some way. He sees this as an opportunity rather than a problem and emphasizes the importance of finding people who already like the product or approach, rather than trying to convince those who don't.

topic-name : Running Startups for 25 Years
description : Phil Libin reflects on his experience of running startups for 25 years and shares that he has never changed anyone's mind from no to yes. Instead, he focuses on finding people who are already interested and getting them excited about the product or approach.
'''
messages_topics = [
    {"role": "user", "content": f"Please extract the topics and descriptions into structured pairs from the following text:\n{topics}\n"}
]
function_name = "structure_topics"
functions = [
    {
        "name": function_name,
        "description": "Extract topics and description into a desired format from a given topics summary of a content",
        "parameters": {
            "type": "object",
            "properties": {
                "topic_pairs": {
                    "type": "array",
                        "items": {
                            "topic": {
                                "type": "string",
                                "description": "The topic name in the given content. Format example: AI in Health Care"
                            },
                            "topic_description": {
                                "type": "string",
                                "description": "The description of the corresponding topic in the given content. Format example: AI in Health Care is widely being applied ..."
                            },
                        },
                    },
                },
            "description": "Array of topic pairs. Each pair contains the topic name and the corresponding description of the topic in the given content",
            "required": ["topic_pairs"],
        },
    }
]
def create_api_call_message(messages, function_name = None, functions = None, temperature = 0):
    if function_name == None and functions == None:
        try:
            print("call messages")
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = messages,
                temperature = temperature,
            )
            response_content = response["choices"][0]["message"]["content"]
            return response_content
        except Exception as exception:
            return exception.args
    elif function_name != None and functions != None:
        try:
            print("call functions")
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = messages,
                temperature = temperature,
                functions = functions,
                function_call = {"name": function_name}
            )
            response_message = response["choices"][0]["message"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            # response_content = response["choices"][0]["message"]["content"] # This likely will to be Null
            # return response_message, function_args
            return function_args
        except Exception as exception:
            return exception.args
    else:
        return "API Call Condition Not Met"

# response, topic_struct = create_api_call_message(messages=messages_topics, function_name=function_name, functions=functions)
response = create_api_call_message(messages=messages_topics, function_name=function_name, functions=functions)

# print(topic)
# print(topic_struct.get("topic_pairs"))
print(response)

# Notes:
## The expect2 but got 1 error is because when the function calling failed, the function only output the error message, so there won't be two outputs: response, func_args instead, the error message
## The message["content"] could be NULL if using function calling, so it won't return anything, so better just to return function_arg (response from function calling)

## Got an error about youtube_transcript_api. Seem it's missing from requirements.txt?
# @dtedesco1 ➜ /workspaces/Podcast_Content_Automation (main) $ /home/codespace/.python/current/bin/python3 /workspaces/Podcast_Content_Automation/test_llm_direct.py
# Traceback (most recent call last):
#   File "/workspaces/Podcast_Content_Automation/test_llm_direct.py", line 3, in <module>
#     from youtube_transcript.functions import get_youtube_transcript
#   File "/workspaces/Podcast_Content_Automation/youtube_transcript/functions.py", line 2, in <module>
#     from youtube_transcript_api import YouTubeTranscriptApi
# ModuleNotFoundError: No module named 'youtube_transcript_api'
# @dtedesco1 ➜ /workspaces/Podcast_Content_Automation (main) $ pip install youtube_transcript_api
# Collecting youtube_transcript_api
#   Obtaining dependency information for youtube_transcript_api from https://files.pythonhosted.org/packages/33/c1/18e32c7cd693802056f385c3ee78825102566be94a811b6556f17783c743/youtube_transcript_api-0.6.1-py3-none-any.whl.metadata
#   Downloading youtube_transcript_api-0.6.1-py3-none-any.whl.metadata (14 kB)
# Requirement already satisfied: requests in /home/codespace/.local/lib/python3.10/site-packages (from youtube_transcript_api) (2.31.0)
# Requirement already satisfied: charset-normalizer<4,>=2 in /home/codespace/.local/lib/python3.10/site-packages (from requests->youtube_transcript_api) (3.2.0)
# Requirement already satisfied: idna<4,>=2.5 in /home/codespace/.local/lib/python3.10/site-packages (from requests->youtube_transcript_api) (3.4)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /home/codespace/.local/lib/python3.10/site-packages (from requests->youtube_transcript_api) (2.0.4)
# Requirement already satisfied: certifi>=2017.4.17 in /home/codespace/.local/lib/python3.10/site-packages (from requests->youtube_transcript_api) (2023.7.22)
# Downloading youtube_transcript_api-0.6.1-py3-none-any.whl (24 kB)
# Installing collected packages: youtube_transcript_api

