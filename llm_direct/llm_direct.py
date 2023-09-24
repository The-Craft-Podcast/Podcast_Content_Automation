import os
import openai

class llm_summarization:
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.openai_api_key

    def get_summary(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text