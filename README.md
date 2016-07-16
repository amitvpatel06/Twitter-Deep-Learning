# Twitter-Deep-Learning


3 NLP models that use deep learning to analysis tweet sentiment!



I wrote 3 deep learning models(for comparison and ensemble purposes) that I trained on the same dataset: [Sentiment 140's set of over 1.5 million tweets](http://help.sentiment140.com/for-students). However, they are set up in way so that they can be easily applied to any text classification task! I used pretrained word vectors from the NLP group at Stanford as well!

# <h1> Model Details: 

The first is a simple RNN (Recurrent Neural Network) that reads a specified number of words in the tweet and then outputs a sentiment probability vector (I kept a uniform size in the number of steps so that I could batch the computations easily). 

The second is a bidirectional LSTM RNN that uses long short-term memory cells and makes a forward pass and a reverse pass of the input word vectors.

The last is a CNN(Convolutional Neural Network) that applies principals from image processing to a tweet's 2 dimensional sentence vector(each row is a word's vector!). This is based on Yoon Kim's paper: http://arxiv.org/abs/1408.5882. 

# <h1> Usage: 

Using these models is easy! I will be posting final trained weights when I have trained them for a longer period of time, but if you would like to apply them to your own datasets, you can also set them up to do that. You will have to write your own code for parsing your input files and feeding them to my dataset constructor(see the utils folder for more information!). Additionally, you can set your model hyperparameters using the config class and you can set your pooling and filter layers(for the CNN) in the Filters class. I have also left sample files for my dataset and the pretrained word vectors so you can get an idea of what the input files look like when reading through my parsing code.

I will shortly be posting details on performance when the models finish training!
