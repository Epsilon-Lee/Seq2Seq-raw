# -*- coding: utf-8 -*-

import sys
import os
import math

class BleuCalculator():
	def __init__(self, cand_path, ref_dir):
		self.cand_path = cand_path
		self.ref_dir = ref_dir

	def calc_bleu(self, cand_lst, multi_ref_lst):
		"""Compute bleu score for cand_lst according to ref_lst

		Argument
		----------
		cand_lst : list
			a list of strings each as a predicted candidate ouputed
			and reformulated from the model

		multi_ref_lst  : list
			a list reference set, each element is a list of strings
			which contains golden references

		Return
		----------
		(bleu1, bleu2, bleu3, bleu4) : tuple
			each represents n-gram bleu precision score

		bleu : float
			the geometric mean of the above tuple

		bp   : float
			brevity penelty

		(hyp_len, ref_len) : tuple
			hypothesis total length and reference total length

		ratio : float
			the ratio of hyp_len and ref_len: hyp_len / ref_len

		Example
		----------
		TODO[glli]

		"""
		cand_count = len(cand_lst)
		ref_count_lst = [len(ref_lst) for ref_lst in multi_ref_lst]

		# check count match
		check_flag = True
		for ref_count in ref_count_lst:
			if cand_count != ref_count:
				check_flag = False
		assert check_flag is True, 'ref cand count not match!'

		# calculate n-gram count_clip for each candidate
		# cand_gram_count = {}
		# ref_gram_count = [{} for i in xrange(len(ref_count_lst))]
		unigram_p, (hyp_len, ref_len) = self.calc_modified_n_gram_precision(
			cand_lst,
			multi_ref_lst,
			ngram=1
		)
		bigram_p, _ = self.calc_modified_n_gram_precision(
			cand_lst,
			multi_ref_lst,
			ngram=2
		)
		trigram_p, _ = self.calc_modified_n_gram_precision(
			cand_lst,
			multi_ref_lst,
			ngram=3
		)
		fourgram_p, _ = self.calc_modified_n_gram_precision(
			cand_lst,
			multi_ref_lst,
			ngram=4
		)
		
		ratio = hyp_len * 1. / ref_len
		bp = 1 if hyp_len > ref_len else math.exp(1. - 1./ratio)
		bleu = bp * math.exp(
			(
			math.log(unigram_p)
			+ math.log(bigram_p)
			+ math.log(trigram_p)
			+ math.log(fourgram_p)
			) / 4.
		)

		return(unigram_p, bigram_p, trigram_p, fourgram_p), bleu, bp, (hyp_len, ref_len), ratio

	def calc_modified_n_gram_precision(
		self,
		cand_lst,
		multi_ref_lst,
		ngram=1
	):
		"""calculate ngram modified precision

		Argument
		----------
		cand_lst : list
			a list of strings each as a predicted candidate ouputed
			and reformulated from the model

		multi_ref_lst : list
			a list reference set, each element is a list of strings
			which contains golden references

		ngram : int
			indicate which ngram statistics to calculate

		Return
		----------
		ngram_modified_p : float
			the modified precision according to the definiton 
			of the original [paper](http://www.aclweb.org/anthology/P02-1040.pdf)

		(hyp_len, ref_len : (int, int)
			candidiate unigram length, reference unigram length

		"""
		# multi_ref_lst = []
		# multi_ref_lst = zip(multi_ref_lst)
		# multi_ref_lst = [tup[0] for tup in multi_ref_lst]
		ref_lst_count = len(multi_ref_lst)
		multi_ref_lst_new = []
		for i in xrange(len(multi_ref_lst[0])):
			ref_combine_lst = []
			for j in xrange(ref_lst_count):
				ref_combine_lst += [multi_ref_lst[j][i]]
			multi_ref_lst_new += [ref_combine_lst]
		multi_ref_lst = multi_ref_lst_new
		del multi_ref_lst_new

		numerator = 0.
		denominator = 0.

		hyp_len = 0
		ref_len = 0
		
		# print('cand_lst length:')
		# print(len(cand_lst))
		# print('multi_ref_lst length:')
		# print(len(multi_ref_lst))
		for cand_sent, ref_sent_lst in zip(cand_lst, multi_ref_lst):
			nu, de, hl, rl = self.calc_sent_statistics(
				cand_sent,
				ref_sent_lst,
				ngram
			)
			# print(ref_sent_lst)
			# print nu, de, hl, rl
			numerator += nu
			denominator += de
			hyp_len += hl
			ref_len += rl

		# print('numerator: %d' % numerator)
		# print('denominator: %d' % denominator)
		ngram_modified_p = numerator / denominator

		return ngram_modified_p, (hyp_len, ref_len)

	def calc_sent_statistics(
		self,
		cand_sent,
		ref_sent_lst,
		ngram
	):
		"""calc ngram count_clip, ngram count, cand/ref length

		Argument
		----------
		cand_sent : str
			the prediceted str object produced by kind of (beam) search
			algorithm

		ref_sent_lst : list
			a list of str objects, each as a reference to cand_sent

		Return
		----------
		nu : int
			numerator to be added to the overall numerator

		de : int
			denominator to be added to the overall denominator

		hl : int
			unigram length of the cand_sent

		rl : int
			unigram length of the reference in ref_sent_lst which is
			the closest to the length of the cand_sent

		"""
		cand_word = cand_sent.split()
		ref_word_lst = [ref_sent.split() for ref_sent in ref_sent_lst]
		ref_count = len(ref_word_lst)
		cand_ngram_count = {}
		ref_ngram_count_lst = [{} for i in xrange(ref_count)]

		for i in range(0, len(cand_word) - ngram + 1):
			gram = cand_word[i : i + ngram]
			gram = " ".join(gram)
			if gram in cand_ngram_count:
				cand_ngram_count[gram] += 1
			else:
				cand_ngram_count[gram] = 1

		for ref_ngram_count, ref_word in zip(ref_ngram_count_lst, ref_word_lst):
			for i in range(0, len(ref_word) - ngram + 1):
				gram = ref_word[i : i + ngram]
				gram = " ".join(gram)
				if gram in ref_ngram_count:
					ref_ngram_count[gram] += 1
				else:
					ref_ngram_count[gram] = 1

		hl = len(cand_word)
		# compute rl as the closest length w.r.t hl
		ref_lens = [len(ref_word) for ref_word in ref_word_lst]
		rl = ref_lens[0]
		len_distance = abs(hl - rl)
		for ref_len in ref_lens:
			if len_distance > abs(hl - ref_len):
				rl = ref_len
				len_distance = abs(hl - ref_len)

		# compute de and nu
		de = len(cand_word) - ngram + 1
 		nu = 0
		for gram, count in cand_ngram_count.iteritems():
			nu_per_gram = 0
			max_ref_count = 0
			for ref_ngram_count in ref_ngram_count_lst:
				if gram in ref_ngram_count:
					ref_count = ref_ngram_count[gram]
					if ref_count > max_ref_count:
						max_ref_count = ref_count
			nu_per_gram = min(count, max_ref_count)
			nu += nu_per_gram

		return nu, de, hl, rl


# ================
# Unit Test
# ================

def unit_test_1():
	cand_path = './debug/de2en_candidates.txt'
	ref_path = './debug/test.en.tok'
	with open(cand_path, 'r') as f_cand:
		cand_lst = f_cand.readlines()
	with open(ref_path, 'r') as f_ref:
		ref_lst = f_ref.readlines()
	cand_lst = [cand_sent.strip() for cand_sent in cand_lst]
	ref_lst = [ref_sent.strip() for ref_sent in ref_lst]
	multi_ref_lst = [ref_lst]
	# print cand_lst
	# print ref_lst
	# print multi_ref_lst

	# create bleucal object
	bleucal = BleuCalculator(cand_path, ref_path)
	ngram_precisions, bleu, bp, (hl, rl), ratio = bleucal.calc_bleu(cand_lst, multi_ref_lst)
	uni_p, bi_p, tri_p, four_p = ngram_precisions
	print("BLEU1: %s BLEU2: %s BLEU3: %s BLEU4: %s BLEU: %s bp: %s hyp_len: %d ref_len: %d ratio: %s" 
		% (format(uni_p, "2.2%"),
			format(bi_p, "2.2%"),
			format(tri_p, "2.2%"),
			format(four_p, "2.2%"),
			format(bleu, "2.2%"),
			format(bp, "0.4f"),
			hl,
			rl,
			format(ratio, "0.4f"))
		)

	# format print
	cand_sent = "i think my personal view is very simple with this slide and sharing access ."
	ref_sent = "i think my personal story is explained very simply on the slide , and it &apos;s access ."
	ref_sent_lst = [ref_sent]
	nu, de, hl, rl = bleucal.calc_sent_statistics(cand_sent, ref_sent_lst, 3)
	print nu, de, hl, rl

if __name__ == '__main__':
	unit_test_1() # passed