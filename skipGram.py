from __future__ import division
import argparse
import pandas as pd

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

import string
from numpy import random
import datetime
import pickle

import os
import math

import spacy
import en_core_web_sm
spacy_nlp = en_core_web_sm.load()

def text2sentences(path):
	sentences = []
	with open(path) as f:
		for i,l in enumerate(f):
			l = l.strip()
			l = spacy_nlp(l)
			l = [token.lemma_.lower() for token in l if not token.is_punct and not token.is_stop and not token.__len__() < 3]
			sentences.append(l)
	return sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
		
		self.w2id, self.freq_dic, nWords = self.word2id(sentences) # word to ID mapping and unigram creation
		self.subsampling_rate = 0.15
		self.vocab = list(self.w2id.keys())
		self.spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
		self.unigram_dic = {k:v**(0.75) for k,v in self.freq_dic.items()} # Papers suggest ^0.75
#		self.subsample_dict = self.subsample(self.freq_dic, sentences)
#		 self.trainset = self.subsample(self.freq_dic, sentences)
		
		self.trainset = set(tuple(i) for i in sentences) # set of sentences
		
		self.nEmbed=100 
		self.negativeRate=negativeRate
		self.winSize = winSize
		self.minCount = minCount # not sure what this minCount is used for
		
		# initialize weight and word matrices
		self.weights_matrix = np.random.randn(len(self.vocab), nEmbed)
		self.word_matrix = np.random.randn(nEmbed, len(self.vocab))
		
		self.loss=[]
		self.accLoss = 0.0
		self.trainWords = 0
	
	def word2id(self, sentences):
		w2id={}
		unigram_dic = {}
		all_words = [item for sublist in sentences for item in sublist]
		currentId = 0
		
		for word in all_words:
			if word not in w2id.keys():
				w2id[word] = currentId # setting incremental count as id
				currentId += 1
			if word not in unigram_dic.keys():
				unigram_dic[word] = 0
			unigram_dic[word] += 1
		return w2id, unigram_dic,len(all_words)
	
#	 def sample(self, omit):
#		 """samples negative words, ommitting those in set omit"""
		
# Random Sampling for the moment
#		 keys_ = set(k for k, v in self.w2id.items() if v not in omit)

	def sample(self, omit):
		"""samples negative words, ommitting those in set omit"""
		
		keys_ = set(k for k, v in self.w2id.items() if v not in omit)
		filtered_negs = {k:self.unigram_dic[k] for k in keys_}
		vals = np.array(list(filtered_negs.values()))/np.sum(np.array(list(filtered_negs.values())))
		rand_neg_words = random.choice(list(filtered_negs.keys()),self.negativeRate, p=vals)
		return [self.w2id[k] for k in rand_neg_words]

	def train(self):
		
		for counter, sentence in enumerate(self.trainset):
			sentence = list(filter(lambda word: word in self.vocab, sentence))

			for wpos, word in enumerate(sentence):
				wIdx = self.w2id[word]
				winsize = np.random.randint(self.winSize) + 1
				start = max(0, wpos - winsize)
				end = min(wpos + winsize + 1, len(sentence))
				
				# instead of computing negative ids for each context word, we compute for each target word by 
				# taking ids outside window size
#				 st = time.time()
				x = [self.w2id[i] for i in sentence[start:end]]
				x.append(wIdx) 
				negativeIds = self.sample(set(x))
				
#				 if counter == 0 and wpos == 0:
#					 print(time.time()-st)
				
				for context_word in sentence[start:end]:
					ctxtId = self.w2id[context_word]
					if ctxtId == wIdx: continue
#					 negativeIds =  self.sample({wIdx, ctxtId})
					self.trainWord(wIdx, ctxtId, negativeIds)
					self.trainWords += 1

			if counter % 1000 == 0:
				print(' > training %d of %d' % (counter, len(self.trainset)))
				self.loss.append(self.accLoss / self.trainWords)
				self.trainWords = 0
				self.accLoss = 0.

	def trainWord(self, wordId, contextId, negativeIds):
		
		alpha_ = 0.01
		vocab_len = len(self.w2id.keys())		
		
		# single forward pass
		
		t = self.weights_matrix[[wordId], :] # (1, nEmbedSize) dimensions
		c = self.word_matrix[:, [contextId]] # (nEmbedSize, 1) dimensions
		n = self.word_matrix[:, negativeIds] # (nEmbedSize, 5) dimensions
		
		
		# need 3 different derivatives - find all derivations in README
		#1. dL/dt
		#2. dL/dc
		#3. dL/dn
		
		dL_dc = -t.T * self.sigmoid(-np.dot(t,c)) # need of size (nEmbed, 1)
		
		dL_dn = np.dot(t.T,self.sigmoid(np.dot(n.T,t.T)).T) # need of size (nEmbed, 5)
				
		diff_c = -c.T * self.sigmoid(-np.dot(t,c)) # need of size (1, nEmbed)
		diff_n = np.dot(self.sigmoid(np.dot(t,n)), n.T) # need of size (1, nEmbed)
		
		dL_dt = diff_c + diff_n # need of size (1, nEmbed)
		
		t = t - alpha_*dL_dt
		c = c - alpha_*dL_dc
		n = n - alpha_*dL_dn
		
		self.weights_matrix[wordId, :] = t
		self.word_matrix[:, contextId] = c[:,0]
		self.word_matrix[:, negativeIds] = n
		
		# defining our loss function
		self.accLoss = -np.log(self.sigmoid(np.dot(t,c))) - np.sum(np.log(self.sigmoid(-np.dot(t,n))))
		
#		 save(word_matrix, path)
#		 save(weights_matrix, path)
#		 print("Loss after {} words is {}".format(self.trainWords, self.accLoss))

	def softmax(self, take_vec):
		probs = list()
		for i in take_vec:
			probs.append(np.exp(i)/np.sum(np.exp(take_vec)))
		return(np.array(probs))
	
	def sigmoid(self, take_vec):
		return(1/(1+np.exp(-1*take_vec)))
	
	def subsample(self, freq_dic, sentences):
		t = self.subsampling_rate
		subsample_dict = {k:(np.sqrt(v/t) + 1) * (t/v) for k,v in freq_dic.items()}
		
		trainset = []
		
		for i,sentence in enumerate(sentences):
#			 temp = []
#			 low = subsample_dict[sentence[0]]
#			 high = subsample_dict[sentence[0]]
#			 for word in sentence:
#				 low = min(subsample_dict[word],low)
#				 high = max(subsample_dict[word],high)
#			 for word in sentence:
#				 if(np.random.uniform(low, high) < subsample_dict[word]):
#					 temp.append(word)
#			 if(len(temp)!=0):
#				 trainset.append(temp)
			
# Using term frequency for calculating probabilites of keeping word in trainset (subsampling) 
			pickNwords = int(0.65*len(sentence)) 
			
			this_sent_dict = {k:v for k,v in subsample_dict.items() if k in sentence}			
			sent_sort_vals = dict(sorted(this_sent_dict.items(),
											 key=lambda item:item[1], reverse=True)[:pickNwords]).values()
			try:
				thres = list(sent_sort_vals)[-1]
			except:
				thres= 0
#				print(i,sentence)
				
			this_sent = []
			for word in sentence:
				if subsample_dict[word] >= thres:
					this_sent.append(word)
			trainset.append(this_sent)
				
		return trainset
	
	def save(self,path):
		# here Im guessing we will be saving a matrix that will serve as a lookup for the word vectors
		# when we want to compute the similarity between words
		with open(path, 'wb') as f:
			pickle.dump(self, f)
		

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		try :
			word1_id = self.w2id[word1]
			word2_id = self.w2id[word2]
			
			w1 = sg.weights_matrix[word1_id,:]
			w2 = sg.weights_matrix[word2_id,:]
			return np.dot(w1, w2)/(np.linalg.norm(w1)*np.linalg.norm(w2))
		except:
			if word1 in self.spacy_stopwords or word2 in self.spacy_stopwords:
				return 0.1
			else:
				return 0.2

	@staticmethod
	def load(path):
		with open(path, 'rb') as f:
			sg = pickle.load(f)
		return sg


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
			# make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))