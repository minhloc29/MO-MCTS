from .openai import OpenAIClient
api_key = "sk-GOCe63jjLF0LZxoy4vemqw"
base_url = "https://mkp-api.fptcloud.com/v1/chat/completions"  # optional if using OpenAI default

# Create client
client = OpenAIClient(
    model="DeepSeek-R1",
    temperature=0.7,
    base_url=base_url,
    api_key=api_key,
)

# Prepare messages in OpenAI chat format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]

# Call the model
response = client._chat_completion_api(messages, temperature=0.7)

# Print the result
print(response[0].message.content)