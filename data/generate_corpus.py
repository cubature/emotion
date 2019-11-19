# -*- coding: UTF-8 -*-
# Based on the original corpus, doing some preprocessing and separating the original into 2 parts (train and test)
# 80% for training, 20% for the test after training
#

import os
import re
import random
from random import randint
import tools as tools

# path of the corpus
path_corpus_original = './text_emotion_original.txt' # the original corpus
path_test_corpus = './test.txt' # the test file to be generated
path_train_corpus = './train.txt' # the train file to be generated


if os.path.exists(path_test_corpus):
	os.remove(path_test_corpus)
with open(path_test_corpus, 'w'):
	pass

if os.path.exists(path_train_corpus):
	os.remove(path_train_corpus)
with open(path_train_corpus, 'w'):
	pass


def write_in_processed_corpus(emotion, sentence):
	number = randint(0, 4)
	if number == 4:  # 1/5 chances to be written into the test file
		with open(path_test_corpus, 'a', encoding='utf-8') as test_corpus:
			test_corpus.write(sentence + ';' + emotion + '\n')
	else:
		with open(path_train_corpus, 'a', encoding='utf-8') as train_corpus:
			train_corpus.write(sentence + ';' + emotion + '\n')


# run through the original corpus 
with open(path_corpus_original, 'r', encoding='utf-8') as corpus_original:
	count_emotion = [0, 0, 0, 0, 0]  # counter of each emotion
	line_number = 0
	for line in corpus_original.readlines():
		line_number += 1
		# first line are headers, useless
		if line_number == 1:
			print(line)
			continue
		# print(line.split(',"'))
		sentence_emotion = line.split(',"')
		# delete the '\n' in the sentence
		sentence = re.sub(r'\\n', ' ', sentence_emotion[3].lower())
		# delete '@<name>', '&amp;'..., éèµ..., "' "or" '", --* not in f**k or sh*t-->cant support *
		sentence = re.sub(r'@\w+\b|&\w+;|\n|[^a-zA-Z1-9\'* ]|(?<!\w)\'|\'(?!\w)|(?<!\w)\*|\*(?!\w)', '', sentence)
		# delete the multiple continuous spaces
		sentence = re.sub(r' {2,}', ' ', sentence).strip()
		emotion = sentence_emotion[1]
		emotion = re.sub(r'\"', '', emotion)
		emotion = tools.emotion_converg(emotion)
		write_in_processed_corpus(emotion, sentence)
		count_emotion[int(emotion)] += 1
	print(count_emotion)

