# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:52:44 2021

@author: jeti8
"""

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]