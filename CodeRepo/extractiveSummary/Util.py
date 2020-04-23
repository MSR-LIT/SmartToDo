from sklearn.feature_extraction.text import CountVectorizer
import spacy
import numpy as np
import re
nlp = spacy.load('en_core_web_sm')

import pdb


def custom_pre_processor(sent):
    sent = sent.replace('\t', ' ').replace('\n', ' ')
    sent = sent.replace('\"', '\'')
    sent = ' '.join(sent.split())  # Replace multiple spaces with single space to avoid tab creation.
    return sent


def spacy_tokenizer(sent):

    doc = nlp(sent, disable = ["ner", "parser", "tagger"])
    lemmas = []
    for tok in doc:
        if tok.is_stop:   #Use corpus specific stop words also. Currently, only spacy stop words are used.
            continue

        tok = tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.lower_

        lemmas.append(tok)

    lemmas = [tok for tok in lemmas if re.match('^[a-z]|[#name]|[?!]', tok)]
    return lemmas



''' During Tokenization, this function identifies named entities and removes those tokens.'''
def spacy_ner_tokenizer(sent):

    doc = nlp(sent, disable = ["parser", "tagger"])
    lemmas = []
    for tok in doc:
        if tok.is_stop:   #Use corpus specific stop words also. Currently, only spacy stop words are used.
            continue

        if tok.ent_type_ == 'PERSON':
            continue

        tok = tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.lower_
        lemmas.append(tok)

    lemmas = [tok for tok in lemmas if re.match('^[a-z]|[#name]|[?!]', tok)]
    return lemmas


def get_sorted_count_terms(corpus, top_k = 5):
    vec = CountVectorizer(preprocessor=custom_pre_processor, tokenizer=spacy_tokenizer).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    sorted_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    word_list = []
    for i in range(min(top_k, len(sorted_freq))):
        word_list.append(sorted_freq[i][0])
    return word_list


def get_embedding_sent_fast_text(sent, embed_dim, embed_func, embed_vocab):

     sent = custom_pre_processor(sent)
     tokens = spacy_ner_tokenizer(sent)
     if len(tokens) == 0:
         embed_mat = np.zeros((1, embed_dim))
     else:
        embed_mat = np.zeros((len(tokens), embed_dim))

     for i in range(len(tokens)):
         # Look up fasttext embeddings vocab for the token.
         tok = tokens[i]
         try:
             embed_mat[i, :] = embed_vocab[tok]
         except:
             continue

     if embed_func == 'mean':
         vec = np.mean(embed_mat, axis = 0)
     elif embed_func == 'max':
         vec = np.max(embed_mat, axis = 0)
     else:
         raise NotImplementedError

     return vec


def computeQueryTokens(doc, max_unique_tokens):

    # vectorize and obtain dictionary
    try:
        vec = CountVectorizer(preprocessor=custom_pre_processor, tokenizer=spacy_tokenizer).fit([doc])
    except:
        return []
    bag_of_words = vec.transform([doc])
    words_freq = [(word, bag_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    sorted_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    vocab_set = set()
    for i in range(min(max_unique_tokens, len(sorted_freq))):
        vocab_set.add(sorted_freq[i][0])

    # Retain words from the original doc if they or their lemmatized version is in the vocab
    final_doc = nlp(doc, disable=["ner", "parser", "tagger"])
    tok_list = []
    for tok in final_doc:
        candidate_tok = tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.lower_
        if candidate_tok in vocab_set:
            tok_list.append(str(tok))

    return tok_list









