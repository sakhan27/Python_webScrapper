
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # for logging purposes
import nltk
import bs4 as bs
import urllib.request
import re
from gensim import corpora, models
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize

##The following code reads the content from a URL
scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Electric_vehicle')
article = scrapped_data.read()
parsed_article = bs.BeautifulSoup(article, 'lxml')

paragraphs = parsed_article.find_all('p')
article_text = ""

for p in paragraphs:
    article_text += p.text


##The following code performs the pre-processing and extracts only NL text
processed_article = article_text.lower().strip() #to remove spaces
processed_article = re.sub('\.', 'STOP', processed_article) #adding a STOP flag where each sentence ends 
processed_article = re.sub('[^a-zA-Z]',' ',processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)


#adding the sentences to a documents array and removing the STOP word
documents = processed_article.strip().split('STOP ')

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]


#Creating the corpus dictionary
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model


# doc_bow = [(0, 1), (1, 1)]
# print(tfidf[doc_bow])  # step 2 -- use the model to transform vectors

#Transform the whole corpus and print out the vector representations
corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print(doc)


lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


lsi_model.print_topics(3)


'''
#You can check the text extracted by executing the following command
# print(processed_article)

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(processed_article)
 
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
 
#print(word_tokens)
# print(filtered_sentence)


frequency = defaultdict(int)
for words in filtered_sentence:
    for word in words:
        frequency[word] += 1

print(frequency['battery'])

texts = [
    [word for word in filtered_sentence if frequency[word] > 1]
    for word in filtered_sentence
]

print(texts)



# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]


# #creating a transformation for tfidf model

# tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

# doc_bow = [(0, 1), (1, 1)]
# print(tfidf[doc_bow])  # step 2 -- use the model to transform vectors

# # corpus_tfidf = tfidf[corpus]
# # for doc in corpus_tfidf:
# #     print(doc)
'''
