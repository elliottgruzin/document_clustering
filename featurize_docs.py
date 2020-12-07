
import glob
import re
import numpy as np
import pickle
import math
# from bert_serving.client import BertClient
import pprint
# from gensim. parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# import nltk
# nltk.download('wordnet')

# bc = BertClient()
## PART 1: get word embeddings

# glove = {}
#
# vecs = np.zeros((400000, 100), dtype=np.float32)
#
# with open('glove.6b/glove.6B.100d.txt','r',encoding='utf8') as dictionary:
#     for i, line in enumerate(dictionary):
#
#         split = line.split()
#         word = split[0]
#         vector = np.asarray(split[1:], "float32")
#         glove[word] = vector
#         vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)
#
# glove['<UNK>'] = np.mean(vecs, axis=0)

# part of solution taken from https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt

## PART 2: clean data, get tfs and dfs

# define dictionaries

tf = {}
df = {}
cleaned_dict = {}
cleaned_sentences = {}
email_ng = {}
ng_set_total = set()
# get files

files = glob.glob('./data/*')

def clean_text(original_text):
    cleaned = re.sub('[!/\'\"(),.:#>-@*\n]',' ', original_text)
    return cleaned

for file in files:

    tf[file] = {}

    with open(file,'r', encoding="utf8", errors='ignore') as input:

        text = input.read()

        # get the newsgroup words and process them

        newsgroup_match = re.search('Newsgroups: (.+)', text).group(1)
        newsgroups = newsgroup_match.split(',')
        ng_words = []
        for group in newsgroups:
            subgroups = group.split('.')
            ng_words.extend(subgroups)
        ng_set = list(set(ng_words))
        email_ng[file] = ng_set
        for element in ng_set:
            ng_set_total.add(element)


        # get subject words

        # subject_match = re.search('Subject: (.+)', text).group(1)
        # subject_match = clean_text(subject_match)
        #
        # # get body text of email
        #
        body_text = re.search(r'\n\n((.|\n)+)$', text).group()

        # combine body text, newsgroup, and subject of email and get words

        cleaned = clean_text(body_text) # + ' ' + subject_match

        cleaned_sentences[file] = cleaned.lower()
        # words = cleaned.lower().split()
        # words_set = set(words)
        #
        # # generate tf and df terms, save cleaned file
        #
        for ng in ng_set:
            try:
                tf[file][ng] += 1
            except KeyError:
                tf[file][ng] = 1

        for ng in ng_set_total:
            try:
                df[ng] += 1
            except KeyError:
                df[ng] = 1
        #
        # cleaned_dict[file] = words
        #
        # cleaned_sentences[file] = words

## compute sentence vectors

# sentence_vectors = bc.encode(cleaned_sentences)

sentence_vectors = {}

# for file in cleaned_dict.keys():
#     # print(file)
#     words = cleaned_dict[file]
#
#     sentence
#
#     word_vectors = []
#     for word in words:
#         try:
#             word_vectors.append(glove[word])
#         except KeyError:
#             word_vectors.append(glove['<UNK>'])
#     word_vectors = np.asarray(word_vectors)
#     tf_idf = np.asarray([math.log(tf[file][word]+1,10)*math.log(300/df[word],10) for word in words])
#     sentence_vectors[file] = np.average(word_vectors, axis = 0, weights = tf_idf)

ng2idx = {ng:idx for idx, ng in enumerate(ng_set_total)}
total_ngs = len(ng_set_total)

for file, ngs in email_ng.items():
    encoding = np.zeros(total_ngs)
    for ng in ngs:
        encoding[ng2idx[ng]] = math.log(tf[file][ng]+1,10)*math.log(300/df[ng],10)
    sentence_vectors[file] = encoding

pprint.pprint(sentence_vectors)

with open('sentence_vectors_ng_tfidf.pkl','wb') as output:
    pickle.dump(sentence_vectors, output)
with open('cleaned_sentences.pkl','wb') as output:
    pickle.dump(cleaned_sentences, output)
