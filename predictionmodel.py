import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords

plt.style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')

# ========================================================
def word_process(df, column):
	# Import stopwords
	stopword_arr = nltk.corpus.stopwords.words('english')

	# Tokenize datafram column
	tokens = df[column].apply(nltk.word_tokenize)

	# Iterate through words and remove stopwords, punctuation, and save as a lower case word
	words = []
	for sent in tokens:
		for word in sent:
			if word.lower() not in stopword_arr and word.lower().isalpha():
				words.append(word.lower())

	return words

# ========================================================
def word_occurrences(words, n):

	# Find frequency distribution of top "n" words
	d = FreqDist(words)
	freq = d.most_common(n)

	# Save the word and its count to two arrays (returned as a list)
	word = []
	count = []
	for tup  in freq:
		word.append(tup[0])
		count.append(tup[1])

	return [word, count]

# ========================================================
def graph_words(word, count, category, color):
	axis = np.arange(len(word))
	plt.bar(axis, count, align='center', color = color, alpha=0.8)
	plt.xticks(axis, word, rotation=45)
	plt.xlabel('Word')
	plt.ylabel('Count')
	plt.title(category)

# ========================================================
def main():
	# Import Main Data
	df = pd.read_csv('2021MCMProblemC_DataSet.csv')
	classes = ['Positive ID', 'Negative ID', 'Unverified']
	colors = ['orange', 'teal', 'red']
	# Separate into different Classes
	df_positive = df.loc[df['Lab Status'] == classes[0]]
	df_negative = df.loc[df['Lab Status'] == classes[1]]
	df_unprocessed = df.loc[df['Lab Status'] == classes[2]]

	# Extract the words (removing punctuation and stopwords)
	words_pos = word_process(df_positive, 'Notes')
	words_neg = word_process(df_negative, 'Notes')
	words_unpro = word_process(df_unprocessed, 'Notes')

	# Count frequency of word occurrences
	n = 50
	word_pos, count_pos = word_occurrences(words_pos, n)
	word_neg, count_neg = word_occurrences(words_neg, n)
	word_unpro, count_unpro = word_occurrences(words_unpro, n)

	# Graph the words as a bar chart
	plt.figure(figsize=(16, 9))
	graph_words(word_pos, count_pos, classes[0], colors[0])
	plt.figure(figsize=(16, 9))
	graph_words(word_neg, count_neg, classes[1], colors[1])
	plt.figure(figsize=(16, 9))
	graph_words(word_unpro, count_unpro, classes[2], colors[2])

	plt.show()

# ========================================================
if __name__ == "__main__":
	main()