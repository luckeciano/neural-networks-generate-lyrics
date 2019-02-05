# neural-networks-generate-lyrics
LSTM-based model for generate music lyrics

# Introduction
This project consists of a learning model, based on deep neural networks, to generate music lyrics.

This project consists of the following steps: 
* Elaborate a script to obtain the database (i.e. lyrics) of a specific style. This script will enter a lyrics site, search for all the artists of a particular musical style, search for all the lyrics of each artist and copy them to that database.
* Model a deep neural network of recurrent architecture and configure its hyperparameters to result in a good generator of music lyrics.
* Train this model in high performance hardware with the database. 
* Generate music lyrics from the trained model - as well as understand and evaluate the training process to improve the model.

# 1 - Web Scraping

In order to obtain sufficient data for the training of the neural network (minimum of 2MB of text), 
which contain enough information to establish a pattern and obtain relevant results, a large amount of music 
lyrics was necessary, where one chose a specific style , in order to maintain the consistency of the dataset.

In order to gather these letters, the Web Scraping technique was used in order to gather the semi-structured information 
of a web page, in a format that meets the training algorithm used.

To this end, we assemble the music lyrics from the following sources: index of the most accessed funk and sertanejo lyrics.

On this page there is a link index of the 1000 most played songs of the style, in addition to the 302 most popular artists.

Initially we saved the addresses for all 1000 songs, and then accessed the index pages of each of the 302 artists, 
incorporating the links to your songs to the array already created.

At the end of the array with enough songs (4386 songs of funk and 5928 of sertanejo) began to read the letters, 
saving all in a single text file, separated by a double line break.
The final files were left with about 5 MB of text each.

# 2 - Neural Network Modeling
Initially, we must understand how the neural network will generate the letter. 
It all relies on supervised learning, and the network functions as a "character classifier". 
Given a sequence of characters, what is the next most likely character to appear, based on training data? 
Although it looks like a fairly simple model, the results are quite impressive.


In this experiment, a deep neural network of recurrent architecture based on long-short term memory (LSTM), 
non-linear activations and dropout was used. The diagram below shows the summary of the chosen model and the number 
of optimized parameters in each layer:

![alt text](https://github.com/luckeciano/neural-networks-generate-lyrics/blob/master/summary.png "Summary")

# Training Procedure, Results and Discussion

See the [presentation](https://github.com/luckeciano/neural-networks-generate-lyrics/blob/master/Rede%20Neural%20compositora%20de%20m%C3%BAsicas.pdf) and the generated samples in this repository (in Portuguese). 
