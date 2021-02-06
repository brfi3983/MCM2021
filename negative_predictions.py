import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer()

plt.style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
class_dist = 1

# ========================================================
def word_process(df, column):
	# Import stopwords
	stopword_arr = nltk.corpus.stopwords.words('english')

	# Tokenize datafram column
	tokens = df[column].apply(str).apply(nltk.word_tokenize)

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
def graph_words(word1, count1, word2, count2, category, color1, color2):

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))
	# USER
	axis = np.arange(len(word1))
	ax1.bar(axis, count1, align='center', color = color1, alpha=0.8)
	ax1.set_xticks(axis)
	ax1.set_xticklabels(word1, rotation = 40, ha='right')
	ax1.set_xlabel('Word')
	ax1.set_ylabel('Count')
	ax1.set_title(f'{category} (User Comments)')

	# LAB
	axis = np.arange(len(word2))
	ax2.bar(axis, count2, align='center', color = color2, alpha=0.8)
	ax2.set_xticks(axis)
	ax2.set_xticklabels(word2, rotation = 40, ha='right')
	ax2.set_xlabel('Word')
	ax2.set_ylabel('Count')
	ax2.set_title(f'{category} (Lab Comments)')

	plt.suptitle('Frequency of Common Words', fontsize=16)
	plt.legend()

	plt.savefig(f'./figures/{category}.png')
# ========================================================
def main():

	# Import Main Data
	df = pd.read_csv('2021MCMProblemC_DataSet.csv')
	classes = ['Positive ID', 'Negative ID', 'Unverified', 'Unprocessed']
	colors = [['orange', 'teal', 'tab:pink', 'tab:brown'], ['red', 'blue', 'tab:purple', 'brown']]

	# Separate into different Classes
	# df_positive = df.loc[df['Lab Status'] == classes[0]]
	# df_negative = df.loc[df['Lab Status'] == classes[1]]
	# df_unverified = df.loc[df['Lab Status'] == classes[2]]
	# df_unprocessed = df.loc[df['Lab Status'] == classes[3]]

	df_lab = df['Lab Comments'].str.lower()
	df_digger = df_lab[df_lab.str.contains('digger', na=False)]
	df_horntail = df_lab[df_lab.str.contains('horntail', na=False)]
	df_sawfly = df_lab[df_lab.str.contains('sawfly', na=False)]
	df_cicada = df_lab[df_lab.str.contains('cicada', na=False)]
	# df_digger = df_lab[df_lab.str.contains('digger', na=False)]
	print(len(df_digger), len(df_horntail), len(df_sawfly), len(df_cicada))
	# print(len(df_positive), len(df_negative), len(df_unverified), len(df_unprocessed))
	exit()
	# Show Class Distribution
	if class_dist == 1:
		plt.figure(figsize=(16,9))
		plt.bar([0, 1, 2, 3], [len(df_positive), len(df_negative), len(df_unverified), len(df_unprocessed)], color = colors[0])
		plt.xticks([0,1,2, 3], classes)
		plt.title('Class Distribution')
		plt.xlabel('Class')
		plt.ylabel('Count')
		plt.savefig('./figures/class_dist.png')
		plt.show()
		exit()

	# Extract the words (removing punctuation and stopwords)
	# USER COMMENTS
	words_pos = word_process(df_positive, 'Notes')
	words_neg = word_process(df_negative, 'Notes')
	words_unver = word_process(df_unverified, 'Notes')

# ========================================================
if __name__ == "__main__":
	main()