"""List the Gemini models your credentials can see: python -m rag.models"""

from google import genai

if __name__ == "__main__":
    client = genai.Client()
    for m in client.models.list():
        name = m.name.split("/")[-1]
        if "gemini" in name or "embedding" in name:
            print(name)
