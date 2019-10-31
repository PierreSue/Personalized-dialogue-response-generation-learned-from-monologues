# Personalized Dialogue Response <br> Generation Learned From Monologues

![image](https://github.com/PierreSue/Personalized-dialogue-response-generation-learned-from-monologues/blob/master/Illstration.png)

This is the implementation code of the [paper - Personalized Dialogue Response
Generation Learned From Monologues](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1696.pdf). We only provide the codes for friends dataset first but you can collect all the dataset you need such as Trump's speech or some lyrics of your favorite singer.

## Introduction
Personalized responses are essential for having an informative and human-like conversation. Because it is difficult to collect a large amount of dialogues involved with specific speakers, it is desirable that chatbot can learn to generate personalized responses simply from monologues of individuals. In this paper, we propose a novel personalized dialogue generation method which reduces the training data requirement to dialogues without speaker information and monologues of every target speaker. In the proposed approach, a generative adversarial network ensures the responses containing recognizable personal characteristics of the target speaker, and a backward SEQ2SEQ model reconstructs the input message for keeping the coherence of the generated responses. The proposed model demonstrates its flexibility to respond to open-domain conversations, and the experimental results show that the proposed method performs favorably against prior work in coherence, personality classification, and human evaluation.

## What has been released in this repository?
We are releasing the following:
* The training code in tensorflow
* Several testing code including cmd_test(test.py, seriestest.py), and file_test(filetest.py) also in tensorflow.
* Manually pruned friends dataset and opensubtitles 

All of the code in this repository works with CPU, and GPU.

## Usage
After collecting all the required training data, you can manipulate the hyperparameters on your own and do the following step if you want to train the whole model and check its performance :
'''
bash run.sh 
bash test.sh
bash seriestest.sh
bash filetest.sh 
'''

Remember to change the GPU setting in main.py if you want.
