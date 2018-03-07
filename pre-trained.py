#import required packages

import pandas as pd
import re
import html.parser
import urllib.request
import urllib.parse
import string

# read the 50 dimensional GloVe vectors from the file
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

# If you have access to 100 or 300-d embeddings, please try it out by changing 50 above, to 100 or 300.
# but then you'll have to play around with the comparing values of cosine similarity later on,
# to get good results of appropriateness, as the current comparing values have been tailored to suit 50-d embeddings


#read 'data_dump' file containing comments coming into the page
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
# Transformed into a list
lst3 = 'i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now'
lst3 = lst3.split(',')

# make a new column 'new1', that will be used to store 50-dimensional word embeddings of the text
df1['new1'] = 0
lst2 = []

#loop through all entries
for i in df1.index:
    lst = []
    words = [j for j in df1.new.loc[i] if j.isalpha()] # for each entry, remove numbers and text such as 'flash32'
    words = [j for j in words if not j in string.punctuation] # remove punctuation
    words = [j for j in words if not j in (lst3 + ['hey', 'hi', 'hello'])] # remove stopwords and words 'hey', 'hello', 'hi', as they have similar embeddings as 'fuck'
    
    for word in words:
        try:
            lst.append(word_to_vec_map[word]) # add the embeddings of each word in 'words' to list 'lst'
        except:
            continue
    # make an array out of the list
	arr = np.array(lst) 
	
    # sum the embeddings of all words in the sentence, to get an embedding of the entire sentence
    # if in the entire sentence, word embeddings weren't available in GloVe vectors, make that sentence into a
    # zero array of length 50
    arrsum = arr.sum(axis = 0) 
    if type(arrsum) != np.ndarray: 
        arrsum = np.array([0] * vector_dim) 
    else:    
        arrsum = arrsum/np.sqrt((arrsum**2).sum()) # normalize the sentence embeddings 
    
    # add the sentence embeddings to the list 'lst2'
    lst2.append(arrsum)
	
# assign sentence embeddings to column 'new1'
df1['new1'] = lst2
# make a new column, 'inappopriate', that says whether given text is appropriate or not
df1['inappropriate'] = 0 # not inappropriate, ie, comment is okay

# for use later on
def cosine_similarity(x, y):
    
    dot = np.dot(x,y)
    # L2 norm of x
    norm_x = np.sqrt(np.sum(x**2))
    # L2 norm of y
    norm_y = np.sqrt(np.sum(y**2))
    cosine_similarity = dot/(norm_x * norm_y)
    
    return cosine_similarity

# loop through data set and lookup the cosine similarites of the sentence with embeddings of different words 
# such as 'appropriate', 'fucking' etc. 
# The values are hyperparameters that have been hard coded after some experimentation, and attempts to strike a balance between recall and 
# precision, although it is impossible to get an exact figure for each without a supervised approach.

for i in df1.index:
    if cosine_similarity(word_to_vec_map['appropriate'], df1.new1.loc[i]) < 0.05:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_to_vec_map['fucking'], df1.new1.loc[i]) > 0.45:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_to_vec_map['fucked'], df1.new1.loc[i]) > 0.5:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_to_vec_map['anal'], df1.new1.loc[i]) > 0.32:
        df1['inappropriate'].loc[i] = 1
    if cosine_similarity(word_to_vec_map['suck'], df1.new1.loc[i]) > 0.52:
        df1['inappropriate'].loc[i] = 1
		
# drop unwanted columns, rename the necessary, and convert to csv
df1.drop(['new', 'new1'], axis = 1).rename(columns = {0 : 'Messages', 'inappropriate' : 'Inappropriate'}).to_csv('Pre-Trained.csv', index = False)
