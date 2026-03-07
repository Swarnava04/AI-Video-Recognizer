import requests;
import ollama;


response=ollama.embed(
    model='bge-m3',
    input=[
        'The sky is blue because of Rayleigh scattering',
        'The quick brown fox jumps over the lazy dog'
    ]
);

embeddings=response.embeddings;

for embedding in embeddings:
    print(embedding[0:5])