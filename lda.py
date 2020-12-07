import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import pickle
from nltk.stem.porter import *

stemmer = SnowballStemmer('english')

# nltk.download('wordnet')

def process_tokens(token):
    # print(token)
    if len(token) > 3 and token not in STOPWORDS:
        stemmed = stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))
        # print(stemmed)
    else:
        # print('sad')
        stemmed = ''
    return stemmed

jar = open('cleaned_sentences.pkl', 'rb')
total_corpus = pickle.load(jar)
jar2 = open('cluster_to_file.pkl','rb')
cluster_to_file = pickle.load(jar2)
n = 0
for cluster, files in cluster_to_file.items():

    if len(files) < 4:
        continue
    print(len(files))
    corpus = [total_corpus[file] for file in files]
    n += 1
    stemmed_sentences = []

    for words in corpus:
        words = words.split()
        new_sentence = ' '.join([process_tokens(word) for word in words]).split()
        stemmed_sentences.append(new_sentence)

    print('done stemming')

    dictionary = gensim.corpora.Dictionary(stemmed_sentences)
    print('built dictionary')
    # dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=50000)
    print('filtered words')
    bow_corpus = [dictionary.doc2bow(doc) for doc in stemmed_sentences]
    print('generated bag of words')
    lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=1, id2word=dictionary, passes=5)
    print('####### CLUSTER {} #######'.format(n))
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

jar.close()
jar2.close()
