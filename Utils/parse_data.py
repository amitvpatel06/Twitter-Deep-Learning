import numpy as np
import csv
import pandas as pd
from collections import defaultdict
from nltk.tokenize.casual import TweetTokenizer


def main():
	print 'hLLO'
	vocab = Vocab()
	print parse_data_set(vocab, "Test.csv", steps=10)

class Vocab(object):
	def __init__(self):
		self.word_to_index = {}
		self.index_to_word = {}
		self.word_freq = defaultdict(int)
		self.total_words = 0
		self.unknown = '<unk>'
		self.add_word(self.unknown, count=0)

	def add_word(self, word, count=1):
		if word not in self.word_to_index:
			index = len(self.word_to_index)
			self.word_to_index[word] = index
			self.index_to_word[index] = word
		self.word_freq[word] += count

	def encode(self, word):
		if word in self.word_to_index:
			return self.word_to_index[word]
		return self.word_to_index['<unk>']

	def decode(self, index):
		return self.index_to_word[index]


def parse_data_iterator(vocab, filename, delimiter=",", steps=10): 
	vocab.add_word('</s>')
	file = open(filename, 'r')
	reader = csv.reader(file, delimiter=delimiter, )
	headers = next(reader)
	list_of_train = []
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
	for row in reader:
		curr = []
		encoded = []
		label = row[1]
		words = tokenizer.tokenize(" ".join(row[3:]))
		for i in range(steps): 
			if i < len(words):
				vocab.add_word(str(words[i]))
				curr.append(words[i])
			else:
				curr.append('</s>')
		for word in curr:
			encoded.append(vocab.encode(word))
		yield label, curr


def parse_data_set(vocab, filename, delimiter=",", steps=10): 
	# filler character for sentences below length of steps
	vocab.add_word('</s>')
	file = open(filename, 'r')
	reader = csv.reader(file, delimiter=delimiter)
	headers = next(reader)
	labels = []
	list_of_train = []
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)
	for row in reader:
		curr = []
		labels.append(row[1])
		words = tokenizer.tokenize(" ".join(row[3:]))
		for i in range(steps): 
			if i < len(words):
				vocab.add_word(str(words[i]))
				curr.append(words[i])
			else:
				curr.append('</s>')
		list_of_train.append(curr)
		# for now we are going to leave out the else case because batching is too slow for unever lenght sentences
	encoded = [[vocab.encode(word) for word in sentence] for sentence in list_of_train ]
	results = {'labels': labels, 'training_examples':list_of_train, 'encoded':encoded}
	parsed_data = pd.DataFrame(data=results)
	print (vocab.index_to_word)
	return parsed_data

def create_embedding_matrix(vocab, wv_filename)
	""" Should have filled vocab when you execute this method!!"""
	word_map = pd.DataFrame.from_dict(vocab.index_to_word)
	word_map['word_vecs'] = np.zeros(len(vocab.index_to_word))
	file = open(filename, 'r')
	reader = csv.reader(file, delimiter=" ")
	processed = set
	for line in reader:
		word = line[0]
		if word in vocab.word_to_index:
			word_map['word_vecs'][vocab.word_to_index[word]] = np.array(line[1:])
	return word_map


if __name__ == "__main__":
    main()