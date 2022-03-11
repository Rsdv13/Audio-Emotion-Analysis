

import nltk                         #dataset
from os import getcwd               #cwd-current working directory for the system
import numpy as np                  #Adds mathematical functions
import pandas as pd                 #splits the data into rows and columns
import re                           #check if a string contains the specified search pattern (regular expression)




#test set from nltk samples
import ssl                          #designed to create secure connection between client and server (Secure Sockets Layer)

try: #check some code for errors
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: #variable assignment or cant append the values
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('twitter_samples')
nltk.download('stopwords')



from nltk.corpus import twitter_samples 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# In[4]:


filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)


# In[5]:


pos_text = twitter_samples.strings('positive_tweets.json')
neg_text = twitter_samples.strings('negative_tweets.json')


# In[6]:


test_pos_set = pos_text[1000:]
train_pos_set = pos_text[:1000]
test_neg_set = neg_text[1000:]
train_neg_set = neg_text[:1000]
x_train = train_pos_set + train_neg_set
x_test = test_pos_set + test_neg_set


# In[7]:


y_train = np.append(np.ones((len(train_pos_set), 1)), np.zeros((len(train_neg_set), 1)), axis=0)
y_test = np.append(np.ones((len(test_pos_set), 1)), np.zeros((len(test_neg_set), 1)), axis=0)


# In[8]:


print("y_train.shape = " + str(y_train.shape))
print("y_test.shape = " + str(y_test.shape))


# In[9]:


#python utils function
import string 

def process_text(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False,        strip_handles=True,reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  
                word not in string.punctuation): 
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


# In[10]:


#python utils function
def build_freqs(tweets, ys):
    # Convert np array to list since zip needs an iterable.
    yslist = np.squeeze(ys).tolist()

    #Count freqs and generate dictionary
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_text(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


# In[11]:


freqs = build_freqs(x_train, y_train)
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))


# In[12]:


print('Positive sentiment: \n', x_train[0])
print('Splitting and processed it: \n', process_text(x_train[0]))


# In[13]:


def sigmoid(z): 
    h = 1/(1+np.exp(-z))
    return h


# In[14]:


def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)
    
    for i in range(0, num_iters):
        z = np.dot(x,theta)
        h = sigmoid(z)
        
        # cost
        J = -(1/m) * (np.dot(np.transpose(y), np.log(h)) +  np.dot(np.transpose(1-y), np.log(1-h)))
        theta = theta - (alpha/m)*(np.dot(np.transpose(x),(h-y)))
        
    J = float(J)
    return J, theta


# In[15]:


def extract_features(text, freqs):
    word_list = process_text(text)
    
    x = np.zeros((1, 3)) 
    
    #bias term 
    x[0,0] = 1 
    
    for word in word_list:
        
        # positive label
        if (word, 1) in freqs:
            x[0,1] += freqs[(word, 1)]
        
        # neg label
        if (word, 0) in freqs:
            x[0,2] += freqs[(word, 0)]
                
    assert(x.shape == (1, 3))
    return x


# In[16]:


X = np.zeros((len(x_train), 3))
for i in range(len(x_train)):
    X[i, :]= extract_features(x_train[i], freqs)

# training labels
Y = y_train

#gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"cost: {J:.8f}.")
print(f"weights: {[round(t, 8) for t in np.squeeze(theta)]}")


# In[17]:


def predict_text(text, freqs, theta):
    x = extract_features(text, freqs)
    y_pred = sigmoid(np.dot(x,theta))
    
      
    return y_pred


# In[18]:


#test
for text in ['happy', 'bad', 'this is awesome', 'great','good good']:
    print( '%s -> %f' % (text, predict_text(text, freqs, theta)))


# In[19]:


def logistic_regression_test(test_x, test_y, freqs, theta):
    y_hat = []
    
    for text in test_x:
        y_pred = predict_text(text, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    
    y_hat = np.asarray(y_hat)
    test_y = np.squeeze(test_y)
    count =0
    for i in range(len(test_y)):
        if (test_y[i] == y_hat[i]):
            count = count+ 1
        else:
            count

    accuracy = count/(len(test_y))
    return accuracy


# In[20]:


test_accuracy = logistic_regression_test(x_test, y_test, freqs, theta)
print(f"Logistic regression model's accuracy = {test_accuracy:.4f}")


# In[21]:


text_sample_1 = "Anjali Sharma spoken to nahin number in about drawing about school number about"
text_sample_2 = 'I am falling from market I am falling from market'
print(process_text(text_sample_1))
y = predict_text(text_sample_1, freqs, theta)
print(y)
if y > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')
    
    
print(process_text(text_sample_2))
y = predict_text(text_sample_2, freqs, theta)
print(y)
if y > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')


# In[ ]:




