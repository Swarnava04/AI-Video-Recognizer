import numpy as np;
from build_embeddings import build_dataframe;
from build_embeddings import create_embeddings;
import faiss;
import pickle;
import ollama;


def build_faiss_index(df, index_path="faiss.index", metadata_path="metadata.pkl"):
    
    # convert embeddings column to numpy array
    embeddings=np.array(df["embeddings"].tolist()).astype("float32");
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings);
    
    dimension = embeddings.shape[1];

    # Create FAISS index (L2 normanlization)
    index=faiss.IndexFlatL2(dimension);

    # Add embeddings
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    #Save metadata
    metadata=df.drop(columns=["embeddings"])
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f);

    print("FAISS index built and saved.")

def vector_search(query, top_k=3):
    df=build_dataframe();
    build_faiss_index(df);
    index=faiss.read_index("faiss.index");


    #Load metadata
    with open("metadata.pkl", "rb") as f:
        metadata=pickle.load(f)
    
    #Embed query
    response=ollama.embed(
        model='bge-m3',
        input=[query]
    )
    query_vector=np.array(response.embeddings).astype('float32');
    # Normalize query
    faiss.normalize_L2(query_vector)

    #search
    distances, indices=index.search(query_vector, top_k);
    results=metadata.iloc[indices[0]]
    return results;


