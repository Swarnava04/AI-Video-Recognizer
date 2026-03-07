import numpy as np;
from build_embeddings import build_dataframe
from build_embeddings import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity;
def cosine_search_1(query, top_k=3):
    df=build_dataframe();
    df['embeddings']=df['embeddings'].apply(lambda x: np.array(x, dtype='float32'))
    embedding_matrix=np.vstack(df['embeddings'].values)  #(N, 1024)
    # embedding_norms=np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    # embedding_matrix_normalized=embedding_matrix/embedding_norms;

    # normalizing the query
    query_matrix=create_embeddings([query])[0].reshape(1, -1);   #(1, 1024)
    # query_norm=np.linalg.norm(query_matrix)
    # query_matrix_normalized=query_matrix/query_norm;

    #cosine similarity
    # simlarities=embedding_matrix_normalized @ query_matrix_normalized
    similarites=cosine_similarity(embedding_matrix, query_matrix);
    df["similarity"]=similarites;

    top_results_df=df.sort_values(by="similarity", ascending=False, axis=0).head(top_k);
    return top_results_df;

    