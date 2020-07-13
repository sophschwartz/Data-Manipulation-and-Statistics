# -*- coding: utf-8 -, pep8*-
"""
Script to be used for Joe's Systematic Review, characterizing the type of
language used in abstracts selected for further analyses.
More from tutorial at:
    https://www.datacamp.com/community/tutorials/wordcloud-python
Input csv with abstract on each line, imported from Reference manager (Zotero).
"""
from collections import Counter
import numpy as np
import nltk
# nltk.download("punkt")
import matplotlib.pyplot as plt
import wordcloud


def listToString(s):
    str1 = " "
    return (str1.join(s))


curr_file = open('test_abstract_bib.csv', encoding="utf8", errors='ignore')

running_list = []
for curr_line in curr_file:
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stripped_words = tokenizer.tokenize(curr_line)
    distinct_words = list(set(stripped_words))
    running_list.extend(distinct_words)

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
running_list_no_sw = [item for item in running_list if item not in stopwords]
word_list = listToString(running_list_no_sw)

# Create word cloud generator object
my_word_cloud = wordcloud.WordCloud(width=800, height=800,
                                    background_color='white',
                                    max_words=20,
                                    stopwords=stopwords,
                                    min_font_size=10).generate(word_list)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(my_word_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# histogram of relevant words
n = 10
word_freq = dict(Counter(running_list_no_sw).most_common(n))
labels, values = zip(*word_freq.items())
# sort your values in descending order
indSort = np.argsort(values)[::-1]
# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.35
plt.bar(indexes, values)
plt.xticks(indexes + bar_width, labels, rotation='vertical')
plt.xlabel('word', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.show()
