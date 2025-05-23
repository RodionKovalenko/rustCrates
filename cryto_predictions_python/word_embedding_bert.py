# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:57:43 2021

@author: jeti8
"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")