# unsupervised-sentiment-analysis

From a large corpus of comments that come into a page, identify and filter out the ones deemed 'inappropriate' or 'toxic', even without the aid of labelled examples.

The models here use GloVe vectors as the basis for identifying toxic comments. There are 2 models, 'pre-trained.py', where GloVe vectors have been used as such, and 'new-trained.py', where GloVe vectors have been trained further on the existing dat set, 'data_dump.txt'. 

The basic structure of the model is as follows:
 1) Translate the commments in the data set to English
 2) For each word in each comment, get the corresponding GloVe vector, and thus get the embeddings for each comment.
 3) Determine the cosine similarity between embeddings of the comment and that of 'toxic' words, and correspondingly set a flag ( = 1), if     it's inappropriate.
 4) In the 'new-trained.py' model, train the existing GloVe vectors further on the data set, and repeat step 4.

The 'weights.npy' file contains the newly trained word vectors after training for 10 epochs (low, since we are training on top of pre trained GloVe vectors), the losses of which can be found in the 'keras_loss.png' image.

The code is heavily documented, and will hopfully dispel all doubts that can creep up.

