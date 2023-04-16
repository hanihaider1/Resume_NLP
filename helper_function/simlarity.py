from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        sentences = [s.split() for s in X]
        self.word2vec_model = Word2Vec(sentences,
                                       vector_size=self.vector_size,
                                       window=self.window,
                                       min_count=self.min_count,
                                       workers=self.workers)
        return self

    def transform(self, X):
        X_transformed = []
        if isinstance(X, str):
            X = [X]
        for s in X:
            if isinstance(s, str):
                vec = [self.word2vec_model.wv[w] for w in s.split() if w in self.word2vec_model.wv]
                vec = np.mean(vec, axis=0) if vec else np.zeros(self.vector_size)
                X_transformed.append(vec)
        if not X_transformed:
            X_transformed = [np.zeros(self.vector_size)]
        return np.array(X_transformed)

def get_similarity(resume_list,job_description_tokenised,holder):
  if holder==1:
    vectorizer = TfidfVectorizer()
    similarity_list = []
    for i in resume_list:
      combined_text = [job_description_tokenised[0], i]
      final_vector = vectorizer.fit_transform(combined_text)
      similarity_list.append(round(cosine_similarity(final_vector)[0][1]*100,2))
  elif holder==2:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    job_tokens = tokenizer.encode(job_description_tokenised[0], add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')

    resume_text = resume_list
    resume_tokens = tokenizer.batch_encode_plus(resume_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')

    job_tokens = tokenizer.batch_encode_plus(job_description_tokenised, padding=True, truncation=True, max_length=128, return_tensors="pt")
    job_embeddings = []
    for i in range(len(job_tokens['input_ids'])):
        job_input_ids = job_tokens['input_ids'][i]
        job_attention_mask = job_tokens['attention_mask'][i]
        job_outputs = model(input_ids=job_input_ids.unsqueeze(0), attention_mask=job_attention_mask.unsqueeze(0))
        job_embedding = job_outputs[1].detach().numpy()
        job_embeddings.append(job_embedding)

    resume_tokens = tokenizer.batch_encode_plus(resume_text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt')
    resume_embeddings = []
    for i in range(len(resume_tokens['input_ids'])):
        resume_input_ids = resume_tokens['input_ids'][i]
        resume_attention_mask = resume_tokens['attention_mask'][i]
        resume_outputs = model(input_ids=resume_input_ids.unsqueeze(0), attention_mask=resume_attention_mask.unsqueeze(0))
        resume_embedding = resume_outputs[1].detach().numpy()
        resume_embeddings.append(resume_embedding)
    

    job_embedding = job_embedding.reshape(job_embedding.shape[0], -1)
    resume_embeddings = np.concatenate(resume_embeddings, axis=0)
    resume_embeddings = resume_embeddings.reshape(resume_embeddings.shape[0], -1)

    similarity_list = cosine_similarity(job_embedding, resume_embeddings)[0]
    #similarity_list = sorted(zip(resume_text, similarity_list[0]), key=lambda x: x[1], reverse=True)
  elif holder==3:
    w2v_transformer = Word2VecTransformer()
    w2v_model = Word2Vec(sentences=resume_list, vector_size=100, window=5, min_count=1, epochs=5)
    w2v_transformer = Word2VecTransformer()
    w2v_transformer.word2vec_model = w2v_model
    similarity_list = []
    
    # preprocess job description
    
    job_desc_vector = w2v_transformer.transform(job_description_tokenised[0])
    
    for selected_resume in resume_list:
        # preprocess selected_resume
        
        selected_resume_vector = w2v_transformer.transform(selected_resume)

        # calculate cosine similarity
        similarity_score = cosine_similarity(selected_resume_vector.reshape(1, -1), 
                                              job_desc_vector.reshape(1, -1))[0][0]

        similarity_list.append(similarity_score)
     

  return similarity_list