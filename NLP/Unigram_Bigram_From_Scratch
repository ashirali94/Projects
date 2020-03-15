### Created by Muhammad Ashir Ali ###
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
from collections import Counter
import math
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            print(word)
            print(freqDict[word])
            if freqDict[word] < 2:

                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus, numberOfSentences, filename):
        print("Parent Inherited")
        self.corpus = corpus
        self.numberOfSentences = numberOfSentences
        self.filename = filename
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        sentence = []
        word = start
        lengthlimit = 0
        while word != end:
            lengthlimit += 1
            sentence.append(word)
            word = self.draw()
            print(word)
            print(lengthlimit)
            if lengthlimit == 100:
                break
        sentence.append(end)
        return sentence
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        prob = 1.0
        for word in sen[1:]:
            prob = prob * self.prob(word)
        print(prob)
        return prob
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        words = [word for sen in corpus for word in sen[1:]]
        log_sum = 0.0
        for word in words:
            prob = self.prob(word)
            if prob!=0:
                log_sum = log_sum + math.log(prob)
        prplxity = math.exp(log_sum/(-len(words)))
        return prplxity
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        print('In generate Sentence Function')
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus, vocab):
        #print("Subtask: implement the unsmoothed unigram language model")
        self.counts_unigram = defaultdict(float)
        self.total = 0.0
        self.vocab = vocab
        self.corpus = corpus
        self.numberOfSentences = 1
        self.filename = 'senfile.txt'
        LanguageModel.__init__(self, corpus, self.numberOfSentences, self.filename)
        self.train(corpus)
    #endddef
    
    ##train the corpus
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start or word == end:
                    continue
                self.counts_unigram[word] += 1.0
                self.total += 1.0
    
    ##probability of model
    def prob(self, word):
        return self.counts_unigram[word]/self.total
    #enddef                
    
    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts_unigram.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
            
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus, vocab):
        #print("Subtask: implement the unsmoothed unigram language model")
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab = vocab
        self.corpus = corpus
        self.numberOfSentences = 1
        self.filename = 'senfile.txt'
        LanguageModel.__init__(self, corpus, self.numberOfSentences, self.filename)
        self.train(corpus)
    #endddef
    
    ##train the corpus
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start or word == end:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
    
    ##probability of model
    def prob(self, word, vocab):
        return (self.counts[word] + 1)/(self.total + len(vocab))
    #enddef                
    
    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word, vocab)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

# Unsmoothed bigram language model
class BigramModel(UnigramModel):
    def __init__(self, corpus, vocab):
        #print("Subtask: implement the unsmoothed unigram language model")
        self.counts_bigrams = defaultdict(float)
        self.unique_bigrams = set()
        self.total = 0.0
        self.corpus = corpus
        self.vocab = vocab
        self.numberOfSentences = 1
        self.filename = 'senfile.txt'
        UnigramModel.__init__(self, corpus, self.vocab)
        
        for sen in corpus:
            prev_word = None
            for word in sen:
                if prev_word != None:
                    self.counts_bigrams[(prev_word,word)] = self.counts_bigrams.get((prev_word,word),0) + 1 
                if word == start or word == end:
                    continue
                else:
                    self.unique_bigrams.add((prev_word,word))
                prev_word = word
        self.unique_bigram_words = len(self.counts_unigram)
    #endddef
    
        
    ##probability of model
    def prob(self, prev_word, word):
        bigram_numerator = self.counts_bigrams.get((prev_word,word),0)
        bigram_denominator = self.counts_unigram.get(prev_word,0)
        
        if bigram_numerator == 0 or bigram_denominator == 0:
            return 0.0
        else:
            return float(bigram_numerator)/float(bigram_denominator)
    #enddef
    
    #sentence of 100 words limit
    def generateSentence(self):
        sentence = []
        word = start
        lengthlimit = 0
        while word != end:
            lengthlimit += 1
            sentence.append(word)
            word = self.draw(word)
            print(word)
            print(lengthlimit)
            if lengthlimit == 100:
                break
        sentence.append(end)
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        prev_word = start
        prob=1.0
        for word in sen[1:]:
            prob = prob * self.prob(prev_word, word)
            prev_word = word
        return prob               
    #enddef
    
    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        prev_word = start
        words = [word for sen in corpus for word in sen[1:]]
        log_sum = 0.0
        for word in words:
            prob = self.prob(prev_word, word)
            prev_word = word
            if prob != 0:
                log_sum = log_sum + math.log(prob)
        prplxity = math.exp(log_sum/(-len(words)))
        return prplxity
    #enddef
    
    # Generate a single random word according to the distribution
    def draw(self, prev_word):
        rand = random.random()
        for word in self.counts_bigrams.keys():
            rand -= self.prob(prev_word, word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
            
    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        print('In generate Sentence Function')
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass



# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        #print("Subtask: implement the unsmoothed unigram language model")
        self.counts_bigrams = defaultdict(float)
        self.unique_bigrams = set()
        self.total = 0.0
        self.corpus = corpus
        self.numberOfSentences = 1
        self.filename = 'senfile.txt'
        UnigramModel.__init__(self, corpus)
        
        for sen in corpus:
            prev_word = None
            for word in sen:
                if prev_word != None:
                    self.counts_bigrams[(prev_word,word)] = self.counts_bigrams.get((prev_word,word),0) + 1 
                if word == start or word == end:
                    continue
                else:
                    self.unique_bigrams.add((prev_word,word))
                prev_word = word
        self.bigram_vocab = len(self.counts_unigram)
    #endddef
    
        
    ##probability of model
    def prob(self, prev_word, word):
        bigram_numerator = self.counts_bigrams.get((prev_word,word),0)
        bigram_denominator = self.counts_unigram.get(prev_word,0)
        
        
        #smoothness
        bigram_numerator = bigram_numerator + 1
        bigram_denominator = bigram_denominator + self.bigram_vocab
        
        if bigram_numerator == 0 or bigram_denominator == 0:
            return 0.0
        else:
            return float(bigram_numerator)/float(bigram_denominator)
    #enddef                
    
    #sentence of 100 words limit
    def generateSentence(self):
        sentence = []
        word = start
        lengthlimit = 0
        while word != end:
            lengthlimit += 1
            sentence.append(word)
            word = self.draw(word)
            print(word)
            print(lengthlimit)
            if lengthlimit == 100:
                break
        sentence.append(end)
        return sentence
    #enddef
    
    def getSentenceProbability(self, sen):
        prev_word = start
        prob=1.0
        for word in sen[1:]:
            prob = prob * self.prob(prev_word, word)
            prev_word = word
        return prob               
    #enddef
    
    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        prev_word = start
        words = [word for sen in corpus for word in sen[1:]]
        log_sum = 0.0
        for word in words:
            prob = self.prob(prev_word, word)
            prev_word = word
            if prob!=0:
                log_sum = log_sum + math.log(prob)
        prplxity = math.exp(log_sum/(-len(words)))
        return prplxity
    #enddef
    
    # Generate a single random word according to the distribution
    def draw(self, prev_word):
        rand = random.random()
        for word in self.counts_bigrams.keys():
            rand -= self.prob(prev_word, word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
            
    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        print('In generate Sentence Function')
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef

#endclass



#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    #find all the rare words for vocabulary
    vocabDict = defaultdict(int)
    for sen in trainCorpus:
	    for word in sen:
	       vocabDict[word] += 1
        #endfor
    #endfor
    vocab = set(vocabDict)

    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    
    unigramModeltrain = UnigramModel(trainCorpus, vocab)
    print("UnigramModel output:")
    print("Probability of \"picture\": ", unigramModeltrain.prob("picture"))
    print("\"Random\" draw: ", unigramModeltrain.draw())
    
