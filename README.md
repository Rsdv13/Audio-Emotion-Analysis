# Audio Sentiment Analysis after a single-channel Multiple Source Separation

The goal of this research work is to use machine learning algorithms to predict the
emotion of a person who is contacting customer care for their product queries. In this
project, the audio is converted into text using Google API. This text sentiment is
trained using multiple supervised learning models like SVM, linear SVM and
Decision tree and the accuracies are compared in which Decision Tree has the best
accuracy. With this the converted text from the audio is tested.


## Motivation

Call Centers or Support Centers in different companies aggregate huge amount of data everyday. From all the conversations, few conversations are not customer satisfactory. Finding the sentiment of the customer helps in determining whether the customer was satisfied with the service or not. 

## Dataset

* Data for source separation was taken from EXOTEL which consists of 300 audio files. Each of these files contains conversation between the customer and agent on various topics.

* RAVDESS dataset was used for classification purpose, it consists of 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. It can be found [here](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) or [here](https://smartlaboratory.org/ravdess/). Dataset consists of different emotions like -  *neutral*, *calm*, *happy*, *sad*, *angry*, *fearful*, *disgust*, *surprised*.

* Twitter dataset of sentiment analysis of text documents is chosen.Assigning a sentiment score to
each phrase and component (-1 to +1)

## Process

We approach this problem in four parts:

**Part 1** - The first phase of our system is segmentation of the audio into parts using Google’s webrtc Voice Activation Detection . The splitted parts of audio are run through Adaptive MAP estimation to find the customer and remove the audio chunk of company caller using the generated super vectors.

**Part 2** – Once the audio file of conversation between the customer and call center agent is
divided into chunks, we pass it through Google API to convert the customer chunk audio to text
which divides our project into two arena.
**Part 3** - Using the RAVDESS emotion dataset the predictive sentiment analysis is run in to multiple classifier models to find the best accuracy of all.
**Part 4** - The text is passed into text sentiment analysis model built using logistic regression
trained with nplk twitter dataset to predict the emotion of the customer
## File description

`convert.py` helps convert *.mp3* audio files into *.wav*

`extract.py` is used to extract files from the RAVDESS dataset and store it together on the basis of emotions

`speaker_diarization.py` is used to separate audio chunks of customer and call centre agent. Once the chunks are separated, only chunks containing customer's voice are considered for sentiment analysis.

`sentiment_classification.py` contains complete code on how various audio features that are extracted to train and test the model along with different architectures created to carry out the experiments.

`text_sentiment_classification.py`
