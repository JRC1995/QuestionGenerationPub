#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import logging
import os
import subprocess
import tempfile

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]

class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def __init__(self):
        self._cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR,
                     'edu.stanford.nlp.process.PTBTokenizer',
                     '-preserveLines', '-lowerCase']
        self._path_to_jar_dirname = os.path.dirname(os.path.abspath(__file__))

    def tokenize(self, sentences):
        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        sentences = "\n".join([sentence.replace("\n", " ") for sentence in sentences])

        # ======================================================
        # tokenize sentence
        # ======================================================
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        p_tokenizer = subprocess.Popen(self._cmd, cwd=self._path_to_jar_dirname,
                                       env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        token_lines, err = p_tokenizer.communicate(sentences.encode('utf-8'))
        lines = token_lines.decode('utf-8').split('\n')

        # ======================================================
        # postprocessing
        # ======================================================
        new_lines = []
        for line in lines:
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                          if w not in PUNCTUATIONS])
            new_lines.append(tokenized_caption)

        return new_lines
