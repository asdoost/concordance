#!/usr/bin/env python3

import re
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
from termcolor import colored
from tabulate import tabulate
from scipy import stats


class concordance():

    # 'The_Old_Man_and_the_Sea.txt'): #'sohrab.txt')
    def __init__(self, word, path='/home/asdoost/Documents/projects/kwic/lob_corpus_untagged/LOB_A.txt'):

        punc = r'[!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~]'
        # Clean page numbers
        self.rmv_num = lambda strg: re.sub('\n+\d\n+', ' ', strg)
        # Clean spaces
        self.rmv_spc = lambda strg: re.sub('\s+', ' ', strg)
        # Clean punctuations
        self.rmv_punc = lambda strg: re.sub(punc, '', strg)
        # Open the file
        raw_txt = open(path, 'r', encoding='utf-8').read()
        # Clean the text
        self.cleaned_txt = self.rmv_spc(self.rmv_punc(raw_txt))
        # Make an indexed dictionary of sentences
        self.tokenized_sents = dict(enumerate(sent_tokenize(raw_txt), 1))
        # Make a frequency word freq_list
        word_list = self.cleaned_txt.lower().split()
        word_set = set(word_list)
        self.freq_list = {}
        for wrd in word_set:
            self.freq_list[wrd] = word_list.count(wrd)
        self.word = re.compile(r"\b{}\b".format(word), re.I)

    def word_list(self, threshold=0):
        """(integer) -> dictionary

        Return a dictionary with words as keys and
        tuples (frequency, Interquartile, Standard Deviation, and Coefficient of Variation) as values

        >>> self.word_list(threshold=5)
        {'join': (17, 32983.5, 22481.4079, 0.8627),
        'country': (38, 40461.5, 21765.2707, 0.669),
        'hospital': (34, 26234.5, 18642.5926, 0.5365), ...
        }
        """
        txt = self.cleaned_txt.lower()
        # Indexing words
        indexed_words = dict(enumerate(word_tokenize(txt), 1))

        indexed_list, stat = {}, {}
        for idx, wrd in indexed_words.items():
            indexed_list[wrd] = indexed_list.get(wrd, [idx]) + [idx]

        for key, val in self.freq_list.items():
            # Standard Deviation
            SD = round(stats.tstd(indexed_list[key]), 4)
            # Coefficient of variation
            CV = round(stats.variation(indexed_list[key]), 4)
            # Interquartile range
            IQR = stats.iqr(indexed_list[key])
            if val >= threshold:
                stat[key] = (val, IQR, SD, CV)
        return stat

    def kwic(self):
        """() -> list of lists

        Return a list of lists containing padding instances.

        >>> self.kwic()
        [['5', 'as', 'Labour', 'M', 'Ps', 'opposed', 'the', 'Government', 'Bill', 'which', 'brought', 'life', 'peers', 'into'],
        ['8', '#', '#', '#', '#', 'Most', 'Labour', 'sentiment', 'would', 'still', 'favour', 'the', 'abolition', 'of'],
        ['34', 'interference', 'by', 'Sir', 'Roy', 's', 'Federal', 'Government', 'in', 'the', 'talks', '#', '#', '#'],
        ['36', 'NEGRO', 'PRESIDENT', 'KENNEDY', 'today', 'defended', 'the', 'appointment', 'of', 'a', 'Negro', 'as', 'his', 'Housing'],
        ['39', 'there', 'is', 'no', 'racial', 'discrimination', 'in', 'Government', 'and', 'State', 'housing', 'projects', '#', '#'],
        ...
        ]
        """
        instances = []
        for idx, sent in self.tokenized_sents.items():
            if self.word.search(sent):
                # Clean the sentence
                sent = self.rmv_punc(sent)
                # Pad the sentence
                lst = (6 * ['#']) + sent.split() + (6 * ['#'])
                # Find all the occurrences of the word
                keys = tuple(i for i, w in enumerate(lst) if self.word.search(w))
                for k in keys:
                    instances.append([str(idx)] + lst[k-6: k+7])
        return instances


    def more(self, start, end=None):
        """(integer, (integer)) -> string

        Return sentences by their indices mentioned as a range in arguments.

        >>> self.more(22, 25)
        IAIN MACLEOD , the Colonial Secretary , denied in the Commons last night that there have been secret negotiations on Northern Rhodesia 's future .
        The Northern Rhodesia conference in London has been boycotted by the two main settlers ' parties — the United Federal Party and the Dominion Party .
        But representatives of Sir Roy Welensky , Prime Minister of the Central African Federation , went to Chequers at the week-end for talks with Mr. Macmillan .
        """
        txt = ''
        if start and end:
            assert start <= end, ValueError("The 'start' argument must be smaller than the 'end' argument.")
            for n in range(start, end):
                txt += self.tokenized_sents[n] + '\n'
        else:
            txt = self.tokenized_sents[start]
        return txt

    def stat(self, threshold=0, start=-6, end=7):
        # Transpose and index the kwic list
        window = {idx: list(tup) for idx, tup in enumerate(zip(*self.kwic()), -7)}
        # Change -7 index by the word "index"
        window['index'] = window.pop(-7)
        # Make a set of nodes
        nodes = set(i.lower() for i in window[0])
        # Remove nodes from the window
        window.pop(0)
        # Make a dictionary of words in the window
        window = {key: {wrd: val.count(wrd) for wrd in set(val)
                        if wrd != '#'} for key, val in window.items()}

        # Make a list for the target span determines by start and end argument
        collocate = list(range(start, 0)) + list(range(1, end))
        # Calculate the frequency of collocates
        collocates = {}
        for span in collocate:
            for wrd, frq in window[span].items():
                wrd = wrd.lower()
                collocates[wrd] = collocates.get(wrd, 0) + frq

        # Apply threshold on collocates
        collocates = {wrd: frq for wrd, frq in collocates.items() if frq > threshold}

        # Extract the contingency values
        obs_frq = {}
        node_freq = sum(self.freq_list[i] for i in nodes)
        for col, frq in collocates.items():
            N = sum(self.freq_list.values())
            uv, u_, _v = frq, node_freq, self.freq_list[col.lower()]
            __ = (N - (uv + u_)) - _v
            obs_frq[col] = [[uv, u_], [_v, __]]

        # Calculate chi-squared for each collocate
        result = {}
        for key, arr in obs_frq.items():
            stat, p, dof, expected = stats.chi2_contingency(arr)
            result[key] = [arr[0][0], expected[0][0], p, stat, dof]
        return result

    def ngram(self, filterby='', n=2, threshold=2):
        """((string), (integre), (integer)) -> dictionary

        Return a dictionary of n-grams as keys and n-grams' frequencies as values.

        >>> self.ngram("go", 2, 5)
        {'to go': 14, 'go to': 5}
        """
        filterby = re.compile(r"\b{}\b".format(filterby), re.I)
        ngrams = {}
        for sent in self.tokenized_sents.values():
            sent = self.rmv_punc(self.rmv_spc(sent.lower()))
            sent = sent.split()
            # Check if word argument is passed to the function
            if filterby != r"\b\b":
                for i in range(len(sent[:-n])):
                    gram = ' '.join(sent[i:i+n])
                    # Check if word exist in the n-gram
                    if filterby.search(gram):
                        ngrams[gram] = ngrams.get(gram, 0) + 1
            # Run if the word  argument is not passed to the function
            else:
                for i in range(len(sent[:-n])):
                    gram = ' '.join(sent[i:i+n])
                    ngrams[gram] = ngrams.get(gram, 0) + 1
        # Apply the threshold to ngrams if it is passed
        if threshold:
            ngrams = {gram: frq for gram, frq in ngrams.items() if frq >= threshold}
        return ngrams


    def show(self, typ, sortedby=1, threshold=0, n=3, filterby=''):

        if typ == 'ngram':
            header = ['n-gram', 'Frequency']
            lst = sorted(self.ngram(filterby, n, threshold).items(), key=lambda x: x[sortedby], reverse=True)

        elif typ == 'kwic':
            # Raise ValueError for wrong sortedby input
            assert sortedby > -6, ValueError("The 'sortedby' argument should be between -6 and 6")
            assert sortedby < 7, ValueError("The 'sortedby' argument should be between -6 and 6")

            lst = self.kwic()

            '''lst = [[l[0], ' '.join(l[1:7]), l[7], ' '.join(l[8:])]
                   for l in lst]                    # Make four columns list
            # Make four columns header
            header = ['index', 'Left', 'Node', 'Right']'''

            # Sort from target/key word to key/target word
            # Sort by positive indices
            if sortedby > 0:
                lst = sorted(lst, key=lambda x: [x[i] for i in range(7 + sortedby, 7, -1)])
            # Sort by negative indices
            elif sortedby < 0:
                lst = sorted(lst, key=lambda x: [x[i] for i in range(7 + sortedby, 7)])
            else:
                lst = sorted(lst, key=lambda x: [x[7]])

            # Make header for the table with 15 columns
            header = ['{:+}'.format(i) for i in range(-7, 7)]
            header[0], header[7+sortedby] = 'Index', 'Collocate'
            header[7] = 'Node'

            # Colorized node and collocate word
            for l in lst:
                l[7] = colored(l[7], 'red', attrs=['bold'])
                l[7+sortedby] = colored(l[7+sortedby], 'blue', attrs=['bold'])

        elif typ == 'stat':
            header = ['Collocate', 'Frequency', 'Expected', 'p-value', 'stat']
            lst = [[x] + y[:4] for x, y in self.stat(threshold).items()]
            lst = sorted(lst, key=lambda x: x[sortedby], reverse=True)

        elif typ == 'word_list':
            header = ['Word', 'Frequency', 'Interquartile Range',
                      'Standard Deviation', 'Coefficient of Variation']
            lst = [[x] + list(y) for x, y in self.word_list(threshold).items()]
            lst = sorted(lst, key=lambda x: x[sortedby], reverse=True)

        # Make table
        table = tabulate(
            lst, headers=header, numalign='center',
            stralign="center", tablefmt="fancy_grid"
        )

        print(table)


    def save(self, filterby='', n=2, threshold=2, p='stat'):
        """((string), (integre), (integer), (string)) -> csvfile

        Return a csvfile of methods in the concordance class.

        >>> x = concordance('go', '/home/user/lob_corpus_untagged/LOB_A.txt')
        >>> x.save('go', 2, 5, 'ngram')
        >>>
        """
        if p == 'kwic':
            head = ['{:+}'.format(i) for i in range(-7, 7)]
            head[0], head[7] = 'Index', 'Node'
            rows = self.kwic()
        elif p == 'ngram':
            head = ['n-gram', 'Frequency']
            rows = self.ngram(filterby='', n=2, threshold=2)
        elif p == 'stat':
            head = ['Collocate', 'Frequency', 'Expected', 'p-value', 'stat', 'dof']
            rows = [[x] + y for x, y in self.stat().items()]

        title = self.word + p + '.csv'
        with open(title, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(head)
            csvwriter.writerows(rows)

    def compare(self, other, filterby='', n=2, threshold=2):
        # Get the two dictionary
        first = self.ngram(filterby, n, threshold)
        second = other.ngram(filterby, n, threshold)
        # Make a set of keys for both dictionary
        setf = set(first.keys())
        sets = set(second.keys())
        # Extract the intersection of two sets
        inter = setf.intersection(sets)
        # Extract the difference of two sets
        diff = setf.difference(sets)
        dic = {}
        for key in inter:
            dic[key] = [first[key], second[key]]
        return dic
