from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

sentence1 = "I love playing chess"
sentence2 = "I work as a Data Scientist"

embeddings = model.encode([sentence1, sentence2])

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])

print(f"Similarity score: {similarity[0][0]}")