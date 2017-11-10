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

		return (unigram_p, bigram_p, trigram_p, fourgram_p),
				bleu,
				bp,
				(hyp_len, ref_len),
				ratio

	def calc_modified_n_gram_precision(
		self,
		cand_lst,
		multi_ref_lst,
		ngram=1
	):
		"""calculate ngram modified precision

		
		"""
