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

    if os.path.exists(cache_path):
        print("Loading from embeddings joblib")
        return joblib.load(cache_path)

    jsons=os.listdir(json_dir)
    chunks_final = []
    chunk_id = 0
    for json_file in jsons:
        with open(f"{json_dir}/{json_file}", "r") as f:
            content=json.load(f);
        
        text_list=[chunk["text"] for chunk in content["chunks"]]
        embeddings=create_embeddings(text_list);

        for i, chunk in enumerate(content["chunks"]):
            chunk["id"]=chunk_id
            chunk_id+=1
            chunk[embeddings]=embeddings[i];
            chunks_final.append(chunk)

    df=pd.DataFrame.from_records(chunks_final);
    joblib.dump(df, cache_path, compress=3)
    return df;