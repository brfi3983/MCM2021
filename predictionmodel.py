import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tokenizer = MWETokenizer()

# plt.style.use('bmh')
plt.style.use('ggplot')
nltk.download('punkt')
nltk.download('stopwords')
class_dist = 0
nltk_f = 0
# ========================================================
def word_process(df, column):

	if nltk_f == 1:
		# Import stopwords
		stopword_arr = nltk.corpus.stopwords.words('english')

		# Tokenize datafram column
		# tokens = df[column].apply(str).apply(nltk.word_tokenize)
		# print(df[column].apply(str))
		tokens = df[column].apply(str).apply(nltk.word_tokenize)
		# exit()
		# tokens1 = []
		# # print(tokens)
		# for n, t in enumerate(tokens):
		# 	if t != []:
		# 		print(t)
		# 		tokens1.append(tokenizer.tokenize(t))
		# 	if n == 50:
		# 		print(tokens1)
		# 		exit()
		# exit()

		# Iterate through words and remove stopwords, punctuation, and save as a lower case word
		words = []
		for sent in tokens:
			for word in sent:
				if word.lower() not in stopword_arr and word.lower().isalpha():
					words.append(word.lower())
		return words, 0
	else:
		df[column] = df[column].fillna('')
		# print(df[column])
		vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2))
		words = vectorizer.fit_transform(df[column])
		# print(vectorizer.get_feature_names())
		# exit()
		return words, vectorizer

# ========================================================
def word_occurrences(words, vectorizer, n):

	if nltk_f == 1:
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
	else:
		count = words.sum(axis=0)
		word = vectorizer.get_feature_names()
		print(count)
		exit()
	return [word[:n], count[:n]]

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

	plt.savefig(f'./figures/{category}.svg')
# ========================================================
def main():

	# Import Main Data
	df = pd.read_csv('2021MCMProblemC_DataSet.csv')
	classes = ['Positive ID', 'Negative ID', 'Unverified', 'Unprocessed']
	colors = [['orange', 'teal', 'tab:pink', 'tab:brown'], ['red', 'blue', 'tab:purple', 'brown']]

	# print(df.head(2).to_latex(index=False))
	# text_file = open("LatexTable.txt", "w")
	# text_file.write(df.head(8).to_latex(index=False))
	# text_file.close()
	# exit()
	# Separate into different Classes
	df_positive = df.loc[df['Lab Status'] == classes[0]]
	df_negative = df.loc[df['Lab Status'] == classes[1]]
	df_unverified = df.loc[df['Lab Status'] == classes[2]]
	df_unprocessed = df.loc[df['Lab Status'] == classes[3]]

	print(len(df_positive), len(df_negative), len(df_unverified), len(df_unprocessed))
	# exit()
	# Show Class Distribution
	if class_dist == 1:
		plt.figure(figsize=(16,9))
		plt.bar([0, 1, 2, 3], [len(df_positive), len(df_negative), len(df_unverified), len(df_unprocessed)], color = colors[0])
		plt.xticks([0,1,2, 3], classes)
		plt.title('Class Distribution')
		plt.xlabel('Class')
		plt.ylabel('Count')
		# plt.savefig('./figures/class_dist.svg')
		plt.show()
		exit()

	# Extract the words (removing punctuation and stopwords)
	# USER COMMENTS
	words_pos, vec_pos = word_process(df_positive, 'Notes')
	words_neg, vec_neg = word_process(df_negative, 'Notes')
	words_unver, vec_unver = word_process(df_unverified, 'Notes')

	# LAB COMMENTS
	words_pos_lab, vec_pos_lab = word_process(df_positive, 'Lab Comments')
	words_neg_lab, vec_neg_lab = word_process(df_negative, 'Lab Comments')
	words_unver_lab, vec_unver_lab = word_process(df_unverified, 'Lab Comments')

	n = 15
	# Count frequency of word occurrences
	# USER COMMENTS
	word_pos, count_pos = word_occurrences(words_pos, vec_pos, n)
	word_neg, count_neg = word_occurrences(words_neg, vec_neg, n)
	word_unver, count_unver = word_occurrences(words_unver, vec_unver, n)

	# LAB COMMENTS
	word_pos_lab, count_pos_lab = word_occurrences(words_pos_lab, vec_pos_lab, n)
	word_neg_lab, count_neg_lab = word_occurrences(words_neg_lab, vec_neg_lab, n)
	word_unver_lab, count_unver_lab = word_occurrences(words_unver_lab, vec_unver_lab, n)

	# Store in an iterable fashion
	print(len(word_pos_lab), len(count_pos))
	print(len(word_neg_lab), len(count_neg))
	# print(len(word_pos_lab), len(word_pos))
	# exit()
	data_words = np.array([[word_pos, word_neg, word_unver], [word_pos_lab, word_neg_lab, word_unver_lab]])
	data_count = np.array([[count_pos, count_neg, count_unver], [count_pos_lab, count_neg_lab, count_unver_lab]])

	# Graphing each Category as a subplot (1,2) -> (user, lab)
	for i in range(data_words.shape[1]):
		w1, w2 = data_words[:, i]
		c1, c2 = data_count[:, i]

		# Graph a single subplot
		graph_words(w1, c1, w2, c2, classes[i], colors[0][i], colors[1][i])

	plt.show()

# ========================================================
if __name__ == "__main__":
	main()