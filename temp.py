from mistralai import Mistral
import os
import dotenv
import os

dotenv.load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
model = "mistral-large-latest"

response = client.chat.complete(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ],
)
print(response.choices[0].message.content)