import os
from openai import OpenAI

api_key = os.getenv('DEEPSEEK_API_KEY')

if not api_key:
	raise RuntimeError('DEEPSEEK_API_KEY is not set')

client = OpenAI(
	api_key = api_key,
	base_url = 'https://api.deepseek.com'
)

response = client.chat.completions.create(
	model = 'deepseek-flash',
	messages = [
		{
			'role': 'system',
			'content': 'You are a helpful AI assistant'
		},
		{
			'role': 'user',
			'content': 'Explain BM25 in one sentence.'
		}
	],
	max_tokens = 500
)

answer = response.choices[0].message.content

print(answer)