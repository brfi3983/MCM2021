import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
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

	df_negative = df.loc[df['Lab Status'] == 'Negative ID']

	df_lab = df_negative[['Notes', 'Lab Comments']] #MAKE SURE ONLY NEGATIVE!!!!!
	df_lab = df_lab.applymap(lambda s:s.lower() if type(s) == str else s)


	# ============== Counting how man people resubmitted (or at least more than once) ==============
	# df_lat = df[['Latitude', 'Longitude']]
	# df_count = np.array(df_lat.pivot_table(index=['Latitude', 'Longitude'], aggfunc='size'))
	# total_submissions = df_count.shape[0]
	# df_count = df_count[df_count > 1]
	# print(f'{df_count.shape[0]/total_submissions}')

	# Separate the most common mistakes and label them as a class
	df_digger = df_lab[df_lab['Lab Comments'].str.contains('digger', na=False)]
	df_horntail = df_lab[df_lab['Lab Comments'].str.contains('horntail', na=False)]
	df_sawfly = df_lab[df_lab['Lab Comments'].str.contains('sawfly', na=False)]
	df_cicada = df_lab[df_lab['Lab Comments'].str.contains('cicada', na=False)]
	df_wasp = df_lab[df_lab['Lab Comments'].str.contains('wasp', na=False)]

	# Replacing y values(adding class label)
	df_digger['Lab Comments'] = df_digger['Lab Comments'].apply(lambda s: 0)
	df_horntail['Lab Comments'] = df_horntail['Lab Comments'].apply(lambda s: 1)
	df_sawfly['Lab Comments'] = df_sawfly['Lab Comments'].apply(lambda s: 2)
	df_cicada['Lab Comments'] = df_cicada['Lab Comments'].apply(lambda s: 3)
	df_wasp['Lab Comments'] = df_wasp['Lab Comments'].apply(lambda s: 4)

	# Concatonate all data together
	data = np.array(pd.concat([df_digger, df_horntail, df_sawfly, df_cicada]))

	# Split into X and y (for y you need to set it as an integer array to avoid errors)
	X, y = data[:, 0], data[:, 1]
	y = y.astype('int32')

	# Split into train and test set after looking at distribution
	print(f'Digger: {len(df_digger)} Horntail: {len(df_horntail)} Sawfly: {len(df_sawfly)} Cicada: {len(df_cicada)}, Wasp: {len(df_wasp)}')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

	# Classifier pipeline (Tokenize -> Frequency of Words -> Linear SVM)
	text_clf = Pipeline([
	('vect', CountVectorizer(stop_words='english')),
	# ('tfidf', TfidfTransformer()),
	# ('clf', MultinomialNB()),
	('clf', SGDClassifier(loss='hinge', penalty='l2',
	alpha=1e-7, random_state=42,
	max_iter=10, tol=None)),
	])

	# Train the classifier and then predict on test set
	text_clf = text_clf.fit(X_train, y_train)
	y_pred = text_clf.predict(X_test)

	# Performance Metrics
	print(metrics.classification_report(y_test, y_pred))
	print(metrics.confusion_matrix(y_test, y_pred))
	# print(f'Total number of negative cases: {len(df[df['Lab Status'] == 'Negative ID'])}')
	# print(len(df_positive), len(df_negative), len(df_unverified), len(df_unprocessed))

# ========================================================
if __name__ == "__main__":
	main()