from operator import itemgetter				#this functionality is NOT needed. It may help slightly, but you can definitely ignore it completely.
import pickle
import numpy as np
#DO NOT CHANGE!


def read_train_file():
	'''
	HELPER function: reads the training files containing the words and corresponding tags.
	Output: A tuple containing 'words' and 'tags'
	'words': This is a nested list - a list of list of words. See it as a list of sentences, with each sentence itself being a list of its words.
	For example - [['A','boy','is','running'],['Pick','the','red','cube'],['One','ring','to','rule','them','all']]
	'tags': A nested list similar to above, just the corresponding tags instead of words. 
	'''						
	f = open('train','r')
	words = []
	tags = []
	lw = []
	lt = []
	for line in f:
		s = line.rstrip('\n')
		w,t= s.split('/')[0],s.split('/')[1]
		if w=='###':
			words.append(lw)
			tags.append(lt)
			lw=[]
			lt=[]
		else:
			lw.append(w)
			lt.append(t)
	words = words[1:]
	tags = tags[1:]
	assert len(words) == len(tags)
	f.close()

	# print(words[1:10])
	# print(tags[1:10])
	return (words,tags)



#NEEDS TO BE FILLED!
def train_func(train_list_words, train_list_tags):

	'''
	This creates dictionaries storing the transition and emission probabilities - required for running Viterbi. 
	INPUT: The nested list of words and corresponding nested list of tags from the TRAINING set. This passing of correct lists and calling the function
	has been done for you. You only need to write the code for filling in the below dictionaries. (created with bigram-HMM in mind)
	OUTPUT: The two dictionaries

	HINT: Keep in mind the boundary case of the starting POS tag. You may have to choose (and stick with) some starting POS tag to compute bigram probabilities
	for the first actual POS tag.
	'''
	dict2_tag_follow_tag= {}
	"""Nested dictionary to store the transition probabilities
    each tag X is a key of the outer dictionary with an inner dictionary as the corresponding value
    The inner dictionary's key is the tag Y following X
    and the corresponding value is the number of times Y follows X - convert this count to probabilities finally before returning 
    for example - { X: {Y:0.33, Z:0.25}, A: {B:0.443, W:0.5, E:0.01}} (and so on) where X,Y,Z,A,B,W,E are all POS tags
    so the first key-dictionary pair can be interpreted as "there is a probability of 0.33 that tag Y follows tag X, and 0.25 probability that Z follows X"
    """
	dict2_word_tag = {}
	"""Nested dictionary to store the emission probabilities.
	Each word W is a key of the outer dictionary with an inner dictionary as the corresponding value
	The inner dictionary's key is the tag X of the word W
	and the corresponding value is the number of times X is a tag of W - convert this count to probabilities finally before returning
	for example - { He: {A:0.33, N:0.15}, worked: {B:0.225, A:0.5}, hard: {A:0.1333, W:0.345, E:0.25}} (and so on) where A,N,B,W,E are all POS tags
	so the first key-dictionary pair can be interpreted as "there is a probability of 0.33 that A is the POS tag for He, and 0.15 probability that N is the POS tag for He"
	"""
	#      *** WRITE YOUR CODE HERE ***    

		
	all_words = []
	for i in train_list_words:
		for j in i:
			all_words.append(j)

	unique_words = list(set(all_words))

	all_tags =['C','D','E','F','I','J','L','M','N','P','R','S','T','U','V','W',',','.',':','-','`',"'",'$','###','#']		

	tag_count ={}
	for i in all_tags:
		tag_count[i] = 0

	for i in train_list_tags:
		for j in i:
			tag_count[j] += 1		

	word_count ={}


	try:
		with open('word_count.pickle','rb') as h:
			word_count = pickle.load(h)

	except:		
		for i in unique_words:
			word_count[i] = all_words.count(i)

		with open('word_count.pickle','wb') as h:
			pickle.dump(word_count,h)	

	word_count['unknownword'] = 0
	word_count['anumber'] = 0
 

	for idx1,i in enumerate(train_list_words):
		for idx2,j in enumerate(i):

			try:
					x=float(j)
					train_list_words[idx1][idx2] = 'anumber'
					word_count['anumber'] += 1
					word_count[j] -= 1
					tag_count[train_list_tags[idx1][idx2]] -= 1
					train_list_tags[idx1][idx2] = 'C'
					tag_count['C'] += 1

			except:
				if word_count[j] == 1 :
					train_list_words[idx1][idx2] = 'unknownword'
					word_count['unknownword'] += 1
					word_count[j] = 0
					tag_count[train_list_tags[idx1][idx2]] -= 1
					train_list_tags[idx1][idx2] = train_list_tags[idx1][idx2]
					tag_count[train_list_tags[idx1][idx2]] += 1

			
				
		with open('tag_count.pickle','wb') as h:
			pickle.dump(tag_count,h)



	for i in all_tags:
		dict2_tag_follow_tag[i] ={}
		for j in all_tags:
			dict2_tag_follow_tag[i][j] = 0



	for idx1,i in enumerate(train_list_tags):
		
		for idx2,j in enumerate(i):

			if idx2 == 0:		
				dict2_tag_follow_tag['###'][j] += 1

			elif idx2 < (len(i)-1):
				dict2_tag_follow_tag[j][i[idx2+1]] += 1

			else:
				dict2_tag_follow_tag[j]['###'] += 1


	for idx1,i in enumerate(all_tags):
		total = 0
		for idx2,j in enumerate(all_tags):
			total += dict2_tag_follow_tag[i][j];
		if total != 0:	
			for idx2,j in enumerate(all_tags):
				dict2_tag_follow_tag[i][j] /= total;

	with open('dict.pickle','wb') as h:
		pickle.dump(dict2_tag_follow_tag,h)


	for idx1,i in enumerate(word_count.keys()):
			
		if word_count[i] != 0:

			dict2_word_tag[i] ={}

			for idx2,j in enumerate(all_tags):
				dict2_word_tag[i][j] = 0


	for idx1,i in enumerate(train_list_words):

		for idx2,j in enumerate(i):


				dict2_word_tag[j][train_list_tags[idx1][idx2]] += 1

	with open('dict2.pickle','wb') as h:
		pickle.dump(dict2_word_tag,h)


	for i in dict2_word_tag.keys():

		for j in dict2_word_tag[i]:

			try:
				dict2_word_tag[i][j] /= tag_count[j]

			except:
				pass

	with open('dict2.pickle','wb') as h:
		pickle.dump(dict2_word_tag,h)



	# END OF YOUR CODE	

	return (dict2_tag_follow_tag, dict2_word_tag)



#NEEDS TO BE FILLED!
def assign_POS_tags(test_words, dict2_tag_follow_tag, dict2_word_tag):

	'''
	This is where you write the actual code for Viterbi algorithm. 
	INPUT: test_words - this is a nested list of words for the TEST set
	       dict2_tag_follow_tag - the transition probabilities (bigram), filled in by YOUR code in the train_func
	       dict2_word_tag - the emission probabilities (bigram), filled in by YOUR code in the train_func
	OUTPUT: a nested list of predicted tags corresponding to the input list test_words. This is the 'output_test_tags' list created below, and returned after your code
	ends.

	HINT: Keep in mind the boundary case of the starting POS tag. You will have to use the tag you created in the previous function here, to get the
	transition probabilities for the first tag of sentence...
	HINT: You need not apply sophisticated smoothing techniques for this particular assignment.
	If you cannot find a word in the test set with probabilities in the training set, simply tag it as 'N'. 
	So if you are unable to generate a tag for some word due to unavailibity of probabilities from the training set,
	just predict 'N' for that word.

	'''

	# print(test_words[1:10])
	# quit()
	output_test_tags = []    #list of list of predicted tags, corresponding to the list of list of words in Test set (test_words input to this function)

	all_tags =['C','D','E','F','I','J','L','M','N','P','R','S','T','U','V','W',',','.',':','-','`',"'",'$','###','#']		

	for idx1,i in enumerate(test_words):

		a = np.zeros((len(all_tags),len(i)) , dtype = float)
		b = np.zeros((len(all_tags),len(i)) , dtype = int)

		for idx2,j in enumerate(i):

			if idx2 == 0:
				for k in range(len(all_tags)):

					try:
						thenum = float(j)
						a[k][idx2] = dict2_tag_follow_tag['###'][all_tags[k]]*dict2_word_tag['anumber'][all_tags[k]]
						b[k][idx2] = k
					except:
						if j in dict2_word_tag.keys():
							a[k][idx2] = dict2_tag_follow_tag['###'][all_tags[k]]*dict2_word_tag[j][all_tags[k]]
							b[k][idx2] = k

						else:
							a[k][idx2] = dict2_tag_follow_tag['###'][all_tags[k]]*dict2_word_tag['unknownword'][all_tags[k]]
							b[k][idx2] = k

			elif idx2 < len(i): 
				for k in range(len(all_tags)):
					x=np.zeros(len(all_tags) , dtype= float)
					for l in range(len(all_tags)):

						try:
							thenum = float(j)
							p = (a[l][idx2-1])*(dict2_tag_follow_tag[all_tags[l]][all_tags[k]])*(dict2_word_tag['anumber'][all_tags[k]])
							x= np.append(x,p)


						except:	
							if j in dict2_word_tag.keys():
								p = (a[l][idx2-1])*(dict2_tag_follow_tag[all_tags[l]][all_tags[k]])*(dict2_word_tag[j][all_tags[k]])
								x= np.append(x,p)
							else:
								p = (a[l][idx2-1])*(dict2_tag_follow_tag[all_tags[l]][all_tags[k]])*(dict2_word_tag['unknownword'][all_tags[k]])
								x= np.append(x,p)


					a[k][idx2] = max(x)
					b[k][idx2] = x.argmax()
						

			
		# for k in range(len(all_tags)):

		# 		a[k][len(i)] = dict2_tag_follow_tag[all_tags[k]]['###']	
		# 		b[k][idx2] = k
			

		tags = []

		for j in range(len(i)-1,-1,-1):
			
			if j == (len(i)-1):
				p = a[: , j]
				tags.append(all_tags[p.argmax()])

			else:
				last_tag = tags[-1]
				r=dict2_tag_follow_tag[last_tag]
				tags.append(max(r , key = lambda i:r[i] ))
				
		tags = tags[::-1]


		output_test_tags.append(tags)


			



	# END OF YOUR CODE

	return output_test_tags



# DO NOT CHANGE!
def public_test(predicted_tags):
	'''
	HELPER function: Takes in the nested list of predicted tags on test set (prodcuced by the assign_POS_tags function above)
	and computes accuracy on the public test set. Note that this accuracy is just for you to gauge the correctness of your code.
	Actual performance will be judged on the full test set by the TAs, using the output file generated when your code runs successfully.
	'''

	f = open('test_public_labeled','r')
	words = []
	tags = []
	lw = []
	lt = []
	for line in f:
		s = line.rstrip('\n')
		w,t= s.split('/')[0],s.split('/')[1]
		if w=='###':
			words.append(lw)
			tags.append(lt)
			lw=[]
			lt=[]
		else:
			lw.append(w)
			lt.append(t)
	words = words[1:]
	tags = tags[1:]
	assert len(words) == len(tags)
	f.close()
	public_predictions = predicted_tags[:len(tags)]
	assert len(public_predictions)==len(tags)

	correct = 0
	total = 0
	flattened_actual_tags = []
	flattened_pred_tags = []
	for i in range(len(tags)):
		x = tags[i]
		y = public_predictions[i]
		if len(x)!=len(y):
			print(i)
			print(x)
			print(y)
			break
		flattened_actual_tags+=x
		flattened_pred_tags+=y
	assert len(flattened_actual_tags)==len(flattened_pred_tags)
	correct = 0.0
	for i in range(len(flattened_pred_tags)):
		if flattened_pred_tags[i]==flattened_actual_tags[i]:
			correct+=1.0
	print('Accuracy on the Public set = '+str(correct/len(flattened_pred_tags)))



# DO NOT CHANGE!
if __name__ == "__main__":
	(words_list_train,tags_list_train) = read_train_file()

	(dict2_tag_tag,dict2_word_tag) = train_func(words_list_train,tags_list_train)

	f = open('test_full_unlabeled','r')

	words = []
	l=[]
	for line in f:
		w = line.rstrip('\n')
		if w=='###':
			words.append(l)
			l=[]
		else:
			l.append(w)
	f.close()
	words = words[1:]
	test_tags = assign_POS_tags(words, dict2_tag_tag, dict2_word_tag)
	assert len(words)==len(test_tags)

	public_test(test_tags)

	#create output file with all tag predictions on the full test set

	f = open('output','w')
	f.write('###/###\n')
	for i in range(len(words)):
		sent = words[i]
		pred_tags = test_tags[i]
		for j in range(len(sent)):
			word = sent[j]
			pred_tag = pred_tags[j]
			f.write(word+'/'+pred_tag)
			f.write('\n')
		f.write('###/###\n')
	f.close()

	print('OUTPUT file has been created')
