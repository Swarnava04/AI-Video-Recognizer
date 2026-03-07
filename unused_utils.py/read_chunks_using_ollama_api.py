import requests;
import ollama;
import json;

# response=requests.post(
#     'http://localhost:11434/api/embeddings',
#     json={
#         "model":'bge-m3',
#         "prompt":"I am learning RAG setup for the first time"
#     }
# )

# with open('embedings_sample.json', 'w') as f:
#     json.dump(response.json(), f, indent=4);

response=requests.post(
    'http://localhost:11434/api/embed',
    json={
        "model":'bge-m3',
        "input":['Hello my name is Swarnava Chakrabarti',
     'I am learning RAG setup for the first time',
     'Learning this via exploring Ollama models']
    }
)

embeddings=response.json()['embeddings'];
print(embeddings);

