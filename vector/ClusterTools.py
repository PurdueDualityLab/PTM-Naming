import os
from openai import OpenAI
import dotenv


def get_embedding(text):
    dotenv.load_dotenv(".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
