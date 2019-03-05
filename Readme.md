<h3>Abstractive summarization using bert as encoder and transformer decoder</h3>

I have used a text generation library called Texar , Its a beautiful library with a lot of abstractions, i would say it to be 
scikit learn for text generation problems.

The main idea behind this architecture is to use the transfer learning from pretrained BERT a masked language model ,
I have replaced the Encoder part with BERT Encoder and the deocder is trained from the scratch.

One of the advantages of using Transfomer Networks is training is much faster than LSTM based models as elimanate sequential behaviour in Transformer models.

Transformer based models generate more gramatically correct  coherent sentences.


<h3> Code </h3>
<pre>
#imports
import tensorflow as tf

#download the texar code and install all the python packages specified in requirement.txt of texar_repo
import sys
!test -d texar_repo || git clone https://github.com/asyml/texar.git texar_repo
if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

</h2>#download the CNN Stories data set and unzip the file</h2>
https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
tar -zxf cnn_stories.tgz


<h2>#preprocessing the cnn data</h2>

from os import listdir
import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h) > 0]
	return story, highlights

# load all stories in a directory
def load_stories(directory):
	stories = list()
	for name in listdir(directory):
		filename = directory + '/' + name
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		stories.append({'story':story, 'highlights':highlights})
	return stories

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned

# load stories
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
f1 = open("stories.txt",'w')
f2 = open("summary.txt",'w')
for example in stories:
  example['story'] = clean_lines(example['story'].split('\n'))
  example['highlights'] = clean_lines(example['highlights'])
  f1.write(" ".join(example['story']))
  f1.write("\n")
  f2.write(" ".join(example['highlights']))
  f2.write("\n")
f1.close()
f2.close()
  
story = open("stories.txt").readlines()
summ = open("summary.txt").readlines() 
train_story = story[0:90000]
train_summ = summ[0:90000]

eval_story = story[90000:91579]
eval_summ = summ[90000:91579]


test_story = story[91579:92579]
test_summ = summ[91579:92579]


with open("train_story.txt",'w') as f:
  f.write("\n".join(train_story))
  
with open("train_summ.txt",'w') as f:
  f.write("\n".join(train_summ))
  
with open("eval_story.txt",'w') as f:
  f.write("\n".join(eval_story))
  
  
with open("eval_summ.txt",'w') as f:
  f.write("\n".join(eval_summ))
  
  
with open("test_story.txt",'w') as f:
  f.write("\n".join(test_story))
  
  
with open("test_summ.txt",'w') as f:
  f.write("\n".join(test_summ))  
  





</pre>
