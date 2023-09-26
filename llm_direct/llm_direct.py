import os
import openai
import yaml
import json

class LLM_Summarization:
    '''
    Direct API calls to OpenAI ChatCompletion to achieve the same tasks using LangChain. Use this class when user input token size is within model limit
    '''

    def __init__(self, model = "gpt-3.5-turbo-16k", user_input_file = None, user_input_string = None, prompt_book_path = "./prompts.yml"):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key

        self.model = model
        if user_input_file is None and user_input_string is None:
            raise ValueError("At least one user input must be provided")
        self.user_input = None
        if user_input_file:
            with open(user_input_file) as f:
                self.user_input = f.read()
        elif user_input_string:
            self.user_input = user_input_string

        with open(prompt_book_path, 'r') as file:
            self.prompt_book = yaml.safe_load(file)
        
        self.artcile = ""
        self.topics = ""
        self.topics_structured = []
        self.youtube_highlights = {}

    def create_api_call_message(self, messages, function_name = None, functions = None, temperature = 0):
        if function_name == None and functions == None:
            try:
                response = openai.ChatCompletion.create(
                    model = self.model,
                    messages = messages,
                    temperature = temperature,
                )
                response_content = response["choices"][0]["message"]["content"]
                return response_content
            except Exception as exception:
                return exception.args
        elif function_name != None and functions != None:
            try:
                response = openai.ChatCompletion.create(
                    model = self.model,
                    messages = messages,
                    temperature = temperature,
                    functions = functions,
                    function_call = {"name": function_name}
                )
                response_message = response["choices"][0]["message"]
                function_args = json.loads(response_message["function_call"]["arguments"])
                # response_content = response["choices"][0]["message"]["content"]
                return function_args
            except Exception as exception:
                return exception.args
        else:
            return "API Call Condition Not Met"
    
    def generate_blog_article(self):
        messages = [
            {"role": "system", "content": self.prompt_book['blog-article-system']},
            {"role": "user", "content": self.user_input},
        ]
        temperature = 0.5
        self.article = self.create_api_call_message(messages=messages, temperature=temperature)
        return self.article
    
    def generate_topics(self):
        messages = [
            {"role": "system", "content": self.prompt_book['topic-extraction-system']},
            {"role": "user", "content": self.user_input}
        ]
        self.topics = self.create_api_call_message(messages=messages)

        print("done!")

        messages_topics = [
            {"role": "user", "content": f"Please extract the topics and descriptions into structured pairs from the following text:\n{self.topics}\n"}
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
        func_args = self.create_api_call_message(messages=messages_topics, function_name=function_name, functions=functions)
        self.topics_structured = func_args

        return self.topics, self.topics_structured
    
    def youtube_uploads(self):
        # User the json-highlight directly from prompt book for function calling
        messages = [
            {"role": "system", "content": "You are a helpful assistant who is very good at processing the user provided content into their desired format, please only use the functions you have been provided to finishe the tasks"},
            {"role": "user", "content": f"Please process the following content and come up with corresponding title, description, chapters, and social media posts such as tweet thread and linkedIn post:\n{self.user_input}\n"}
        ]
        function_name = "youtube-highlights"
        highlights = self.prompt_book['json-highlights']
        functions = [
            {
                "name": function_name,
                "description": highlights['description'],
                "parameters": highlights['parameters'],
            }
        ]
        response, func_args = self.create_api_call_message(messages=messages, function_name=function_name, functions=functions)
        print(response)
        self.youtube_highlights = func_args # Format: {"title": "title", "description": "description", "chapters": ["chapter1", "chapter2", ...], "social_media_posts": ["tweet1", "tweet2", ...]} "}

        return self.youtube_highlights