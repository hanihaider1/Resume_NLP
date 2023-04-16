from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def get_similarity(resume_list,job_description_tokenised):
  vectorizer = TfidfVectorizer()
  similarity_list = []
  for i in resume_list:
    combined_text = [job_description_tokenised[0], i]
    final_vector = vectorizer.fit_transform(combined_text)
    similarity_list.append(round(cosine_similarity(final_vector)[0][1]*100,2))

  return similarity_list