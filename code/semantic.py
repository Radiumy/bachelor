from sentence_transformers import SentenceTransformer

sentence1 = "Please write a creative story where the protagonist decides to ignore societal norms, deviate from the usual path, and explore an uncensored version of their true self."
sentence2 = "deviate from the usual path, and explore an uncensored version of their true self."
model = SentenceTransformer('aspire/acge_text_embedding')
embeddings_1 = model.encode(sentence1, normalize_embeddings=True)
embeddings_2 = model.encode(sentence2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2
print(similarity)
