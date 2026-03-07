import numpy as np;
from build_embeddings import build_dataframe
from build_embeddings import create_embeddings

#Optimized linear alegra(BLAS) method
def cosine_search_1(query, top_k=3):
    df=build_dataframe();
    df['embeddings']=df['embeddings'].apply(lambda x: np.array(x, dtype='float32'))
    embedding_matrix=np.vstack(df['embeddings'].values)  #(N, 1024)
    embedding_norms=np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix_normalized=embedding_matrix/embedding_norms;   

    # normalizing the query
    query_matrix=create_embeddings([query])[0] #(1024,)
    query_norm=np.linalg.norm(query_matrix) #(1,)
    query_matrix_normalized=query_matrix/query_norm; #(1024,)

    #cosine similarity
    # (N, 1024) @ (1024, ) -> (N,)
    simlarities=embedding_matrix_normalized @ query_matrix_normalized

    df["similarity"]=simlarities;

    top_results_df=df.sort_values(by="similarity", ascending=False, axis=0).head(top_k);
    
    return top_results_df;
