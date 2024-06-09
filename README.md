# Word2Vec Algorithm for word embeddings

## Getting Started

This project implements Word2Vec algorithm (The Continuous Bag Of Words and not the skipgram model) from scratch and by using the gensim library.
### Tech Stack

* PyTorch 
* gensim
* torchtext

## Overview

Some keypoints of this project:

* Word2vec is a technique in natural language processing (NLP) for obtaining vector representations of words.
* These vectors capture information about the meaning of the word based on the surrounding words.
* In particular, words which appear in similar contexts are mapped to vectors which are nearby as measured by cosine similarity.
* Word2vec can utilize either of two model architectures: either CBOW or SKIPGRAM.
  
## CBOW Architecture used to create word embeddings:

![image](https://github.com/AkshayKulkarni3467/Word2Vec/assets/129979542/ebbef051-65f6-4e49-a658-ecadc31b0b8d)


## SKIPGRAM Architecture used to create word embeddings:

![image](https://github.com/AkshayKulkarni3467/Word2Vec/assets/129979542/3f8f5020-4c59-41c8-a3dc-1d5c397be48e)


## How the context of words is captured:

![image](https://github.com/AkshayKulkarni3467/Word2Vec/assets/129979542/d35d4ebf-3a3a-484a-abe4-1f3b22ca7cc0)
