import ollama;
import requests;
import os;
import json;
import pandas as pd;
import numpy as np;

def create_embeddings(text_list):
    response=ollama.embed(
    model='bge-m3',
    input=text_list
    );
    embeddings=response.embeddings;
    return embeddings;





curr_file=os.path.abspath(__file__);
# print(curr_file);

curr_dir=os.path.dirname(os.path.abspath(__file__));
# print(curr_dir);

parent_dir=os.path.dirname(curr_dir);
# print(parent_dir)

json_dir=os.path.join(parent_dir,"json_files");
# print(json_dir);

jsons=os.listdir(json_dir);
chunks_final=[]
chunk_id=0
# print(jsons);
for json_file in jsons:
    with open(f"{json_dir}/{json_file}", 'r') as f:
        content=json.load(f);
    # converting to embeddings for each file all at once
    text_list=[chunk['text'] for chunk in content['chunks']];
    embeddings=create_embeddings(text_list);

    chunk_content={}
    for i, chunk in enumerate(content["chunks"]):
        chunk["id"]=chunk_id;
        chunk_id+=1
        chunk["embeddings"]=embeddings[i];

        chunks_final.append(chunk);
        # print(chunk);
    # break;

df=pd.DataFrame.from_records(chunks_final);

#incoming query
incoming_query=input("Ask a question: ")
question_embedding=create_embeddings([incoming_query])[0];


# cosine similarity

#Convert stored embeddingg
df['embeddings']=df['embeddings'].apply(lambda x: np.array(x, dtype='float32'))
embedding_matrix=np.vstack(df['embeddings'].values)  #(N, 1024)

# Normalize stored embeddings
embeddings_norms=np.linalg.norm(embedding_matrix, axis=1, keepdims=True) #(N,1)
embedding_matrix_normalized=embedding_matrix/embeddings_norms;
# print(embedding_matrix_normalized);

# Normalize query embedding
question_embedding=np.array(question_embedding, dtype="float32")
question_norm=np.linalg.norm(question_embedding)
question_embedding_normalized=question_embedding/question_norm;

#Compute Similarity
similarites=embedding_matrix_normalized @ question_embedding; # @ is matrix multiplication
# print(similarites.shape);


df['similarity']=similarites;

top_result=3;

top_result_df=df.sort_values(by='similarity',ascending=False, axis=0).head(top_result);


print(top_result_df[['number', 'title', 'similarity']])




