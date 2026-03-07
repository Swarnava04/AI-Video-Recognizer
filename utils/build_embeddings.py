import ollama;
import os;
import json;
import pandas as pd;
import numpy as np;
import joblib;

def create_embeddings(text_list):
    response=ollama.embed(
        model='bge-m3',
        input=text_list
    )
    return response.embeddings;

def build_dataframe():
    curr_dir=os.path.dirname(os.path.abspath(__file__));
    parent_dir=os.path.dirname(curr_dir)
    
    json_dir=os.path.join(parent_dir, "json_files")
    cache_path=os.path.join(parent_dir, "chunks_embeddings.joblib")
    metadata_path=os.path.join(parent_dir, "embeddings_metadata.json")
    jsons=sorted(os.listdir(json_dir))
    if os.path.exists(cache_path) and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata=json.load(f);
        
        if (metadata["json_files"]==jsons):
            print("Loading from cache_embeddings....")
            return joblib.load(cache_path)

    chunks_final = []
    chunk_id = 0
    for json_file in jsons:
        with open(f"{json_dir}/{json_file}", "r") as f:
            content=json.load(f);
        
        text_list=[chunk["text"] for chunk in content["chunks"]]
        embeddings=create_embeddings(text_list);

        for i, chunk in enumerate(content["chunks"]):
            new_chunk=chunk.copy()
            new_chunk["id"]=chunk_id
            chunk_id+=1
            new_chunk["embeddings"]=embeddings[i];
            chunks_final.append(new_chunk)

    df=pd.DataFrame.from_records(chunks_final);
    joblib.dump(df, cache_path, compress=3)

    with open(metadata_path, "w") as f:
        json.dump({"json_files": jsons}, f);
    return df;