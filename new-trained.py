#import required packages
import pandas as pd
import re
import html.parser
import urllib.request
import urllib.parse
import string

def read_glove_vecs(file):
    with open(file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt') # replace file path with your location for 50-d embeddings

# size of embeddings.
vector_dim = 50

#read 'data_dump' file
df1 = pd.read_csv('data_dump.txt' ,sep = '\t', header = None)

# code for translating text to english, inspired by code shared on GitHub
agent = {'User-Agent':
"Mozilla/4.0 (\
compatible;\
MSIE 6.0;\
Windows NT 5.1;\
SV1;\
.NET CLR 1.1.4322;\
.NET CLR 2.0.50727;\
.NET CLR 3.0.04506.30\
)"}


def unescape(text):
    parser = html.parser.HTMLParser()
    return (parser.unescape(text))


def translate(sent_to_translate, to_language="auto", from_language="auto"):
   
    sent_to_translate = urllib.parse.quote(sent_to_translate)
    link = "https://translate.google.com/m?hl={}&sl={}&q={}".format(to_language, from_language, sent_to_translate)
    request = urllib.request.Request(link, headers=agent)
    data = urllib.request.urlopen(request).read().decode("utf-8")
    translation = re.findall(r'class="t0">(.*?)<', data)
    if (len(translation) == 0):
        result = ''
    else:
        result = unescape(translation[0])
    return result

#pass all entries from data set into translator, and add them to list 'lst'
# and print out no. completed after every 500 samples
j = 0
lst = []
for i in df1[0]:
    j+=1
    lst.append(translate(i, to_language = 'en'))
    if j%500 == 0: print ('{} entries completed'.format(j))

# make a new column in the dataframe that shows the translated text
df1['new'] = lst

#separate entries like '2am' to '2 am', ie, add space b/w number and text after making it lower case.
df1.new = df1.new.map(lambda x: (re.sub('(\d)([a-zA-Z]+)', r'\1 \2', x.lower())))

#transforms the text into a list, with words separated from punctuations (just doing a text.split() would have tokens where
#punctuations immediately following a character would not be separated from the character)
df1.new = df1.new.map(lambda x: re.findall('[\w]+|[,;.?!#&]', x))

# the common stop words in the english language. I didn't have access to nltk.stopwords, so I did it this way, as 
# the stop words wore easily available on a google search
# transformed into a list
lst3 = 'i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now'
lst3 = lst3.split(',')


# for use later on
def cosine_similarity(x, y):
    
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(x,y)
    # Compute the L2 norm of u (≈1 line)
    norm_x = np.sqrt(np.sum(x**2))
    # Compute the L2 norm of v (≈1 line)
    norm_y = np.sqrt(np.sum(y**2))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot/(norm_x * norm_y)
    
    return cosine_similarity


# train your own word embeddings on top of pre-trained GLoVe vectors

# training our own word vectors on our own corpus helps find further similarities between the words

# TO MAKE DATA INTO THE FORMAT REQUIRED BY KERAS:
# make the entire text from the data set into one huge corpus, with each example separated from the rest
# by 4 spaces (4 has been used because later on, during the generation of skipgrams, a window size of 4 wil be used
# to identify context words of a target word)
# if the spaces weren't added, in the huge corpus, if one sentene ends in a positive way like 'good', and the 
# next sentence started with 'fuck' then 'good' would turn into a context word for 'fuck', thereby pushing their embeddings 
# towards each other, making their cosine similarity increase

lst4 = []
for i in df1.index:
    lst4+=df1.new.loc[i] + [(' '), (' '), (' '), (' ')]

# do similar pre-processing to remove punctuations and stop words
[lst4.remove(i) for i in lst4 if i in string.punctuation]
[lst4.remove(i) for i in lst4 if i in lst3]

# make it into a series to get unique entries of the huge corpus
# (set() could have been used, but it generates an unordered set of words, which changes the index being assigned to each word
# each time the code is run.)
series = pd.Series(lst4)
dic = {}

# make a dictionary of words with corresponding indexes
# Here, the index of a particular word remains same every time the code is run
for index,word in enumerate(series.unique()):
    dic[word] = index 
# transform the huge corpus into corresponding indexes
for i,j in enumerate(lst4):
    lst4[i] = dic[j]
	
# TRAIN NEW WORD EMBEDDINGS ON CORPUS
# import necessary keras modules
from keras.preprocessing import sequence
from keras.layers import Dot, Reshape, Dense
from keras.models import Model

# size of the vocabulary ,ie, no. of unique words in corpus
vocab_size = len(dic) + 1

# sampling table used to make skipgrams, so that in the negative samples, the most common words are assigned a lower weight
sampling_table = sequence.make_sampling_table(vocab_size)

# make the skipgrams from the corpus, with a window size of 4 for the context words and use samples generated by previous line
# returns tuple of (target word, context word) and associated label of the tuple (1 for whether context in tuple is in fact 
# context for the word in the actual data set, 0 otherwise) 
tuples, labels = sequence.skipgrams(lst4, vocab_size, window_size=window_size, sampling_table=sampling_table)

# extract the target and context words and convert them into arrays (bear in mind that target and context words are 
# now represented by their corresponding indexes from 'dic' dictionary)
target_word, context_word = zip(*tuples)
word_target = np.array(target_word, dtype="int32")
word_context = np.array(context_word, dtype="int32")

# make a new embedding matrix. The pre-trained GloVe vectors are going to be loaded into this matrix
# initialise with zeros
embedding_matrix = np.zeros((vocab_size, vector_dim))

# corresponding to the index of each row of embedding matrix, fill in the values of 50 dimensional word embedddings
for word,index in dic.items():
    try:
        embedding_matrix[index,:] = word_to_vec_map[word]
    except:
        continue # if word is not present in GloVe vectors, that index position is already filled with zeros, as we had initialized
                 # all rows to zero in the first place

            
# START BUILDING THE KERAS MODEL FOR TRAINING
input_target = Input((1,))
input_context = Input((1,))

# make a Keras embedding layer of shape (vocab_size, vector_dim) and set 'trainable' argument to 'True'
embedding = Embedding(input_dim = vocab_size, output_dim = vector_dim, input_length = 1, name='embedding', trainable = True)

# load pre-trained weights(embeddings) from 'embedding_matrix' into the Keras embedding layer
embedding.build((None,))
embedding.set_weights([embedding_matrix])

# run the context and target words through the embedding layer
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)

# compute the dot product of the context and target words, to find the similarity (dot product is usually a measure of similarity)
dot = Dot(axes = 1)([context, target])
dot = Reshape((1,))(dot)
# pass it through a 'sigmoid' activation neuron; this is then comapared with the value in 'label' generated from the skipgram
out = Dense(1, activation = 'sigmoid')(dot)

# create model instance
model = Model(input = [input_context, input_target], output = out)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

# fit the model, default batch_size of 32
# running for 10 epochs seems to generate good enough results, although running for more iterations may improve performance further
model.fit(x = [word_target, word_context], y = labels, epochs = 10,)

# get the new word embeddings and save the new array of shape (vocab_size, vector_dim), to 'word_vecs'.
# here, the second layer of the model is the embedding layer, as can be seen from the index '[2]'
word_vecs = model.layers[2].get_weights()[0]

# NOTE: since 'make_sampling table' and 'skipgrams' select context and target words randomly, each time the model is trained,
# it will give rise to slightly different word embeddings that ultimately will give poor results when used to find cosine similarity
# with the existing comparing values, such as .45 for 'fuck' etc

# So, in order to obtain results good results with existing comparison values, load the embeddings of shape
# (vocab_size, vector_dim) from 'weights.npy' file

# please make sure that you have loaded the 'weights.npy' file into the current directory
word_vecs = np.load('weights.npy')

# loop through data set and lookup the cosine similarites of the sentence with embeddings of different words 
# such as 'appropriate', 'fucking' etc. 
# The values have been hard coded after a lot of experimentation, and attempts to strike a balance between recall and 
# precision, although it is impossible to get an exact figure for each without a supervised approach.

df1['new1_trained'] = 0
lst2 = []
for i in df1.index:
    lst = []
    words = [j for j in df1.new.loc[i] if j.isalpha()] 
    words = [j for j in words if not j in string.punctuation]
    words = [j for j in words if not j in lst3 + ['hello', 'hi', 'hey']]
    
    for word in words:
        try:
            lst.append(word_vecs[dic[word]]) # new embeddings accessed through index of particular word in 'dic'
        except:
            continue
    arr = np.array(lst)
    arrsum = arr.sum(axis = 0)
    if type(arrsum) != np.ndarray:
        arrsum = np.array([0] * vector_dim)
    else:    
        arrsum = arrsum/np.sqrt((arrsum**2).sum())
        
    lst2.append(arrsum)
	
df1['new1_trained'] = lst2
df1['inappropriate'] = 0
for i in df1.index:
    if cosine_similarity(word_vecs[dic['appropriate']], df1.new1_trained.loc[i]) < -0.1:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_vecs[dic['fuck']], df1.new1_trained.loc[i]) > 0.45:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_vecs[dic['suck']], df1.new1_trained.loc[i]) > 0.45:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_vecs[dic['sex']], df1.new1_trained.loc[i]) > 0.4:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_vecs[dic['horny']], df1.new1_trained.loc[i]) > 0.25:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_vecs[dic['anal']], df1.new1_trained.loc[i]) > 0.35:
        df1['inappropriate'].loc[i] = 1

# rename columns, drop unnecessary ones, convert to csv file
df1.rename(columns = {0: 'Messages', 'inappropriate' : 'Inappropriate'}).drop(['new', 'new1', 'new1_trained'], axis = 1).to_csv('New-Trained.csv', index = False)
