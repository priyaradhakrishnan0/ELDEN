#Class for calculation PMI (pointwise mutual information)
#original source - https://github.com/malllabiisc/mall-main/blob/master/src/Allen-AI/Solvers/PMI/pmi_index.py
#modified - August 2016 by priya.r

import time
import sys
import cPickle as pickle
import itertools
import os
from os.path import join
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy import io
import threading


''' global variables '''
stopwords = set()
vocab = {}
co = np.zeros((1, 1))  # co-ocurrence matrix, dummy initialization to begin with
stem_cache = dict()
cooccurence_window = 20
start_time = time.time()
''' paths and directories'''
stopwords_file = '/scratch/home/priya/stopwords.txt'

#Thread Variables
maxThreads = 4 #max threads running at a time
threadLock = threading.Lock()
threadCount = 0
threadLock = threading.Lock()

class fileThread (threading.Thread):
    def __init__(self, threadID, fName, fCount):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.fName = fName
        self.fCount = fCount
    def run(self):
        print "Starting file thread : " + str(self.threadID)
        populate(self.fName)
        print "Exiting file thread : " + str(self.threadID)

# given a string, it preserves only 'a' - 'z' characters
def clean(s):
    l = ""
    for i in range(len(s)):
        ch = s[i]
        o = ord(ch)
        if (o <= ord('z') and o >= ord('a')):
            l += ch
            continue
        else:
            l += ' '
    return l


# given a list of unigrams, returns the list of bigrams present in each window
def get_bigrams(l, window_size):
    list_bigrams = set()
    i = 0
    while i < len(l):
        j = i + 1
        temp = set()
        while j < min(i + window_size, len(l)):
            #if l[j] not in stopwords and l[j-1] not in stopwords:
            if l[j-1] + "_" + l[j] in vocab:
                temp.add(l[j-1] + "_" + l[j])
            j += 1
        list_bigrams.update(temp)
        i += 1
    return list(list_bigrams)


# given a list of unigrams, returns the list of skipgrams present per window
def get_skipgrams(l, window_size):
    list_bigrams = set()
    i = 0
    while i < len(l):
        left = i
        right = min(i + window_size, len(l))
        list_bigrams.update(get_small_skipgrams(l, left, right))
        i += 1
    return list(list_bigrams)


# a helper function to generate combinations of bigrams in a given list 'l'
# starting from left and ending at right
def get_small_skipgrams(l, left, right):
    i = left
    bigrams = set()
    while i < right:
        j = i+1
        while j < right:
            #if l[i] not in stopwords and l[j] not in stopwords:
            if l[i] + "_" + l[j] in vocab:
                bigrams.add(l[i] + "_" + l[j])
            j += 1
        i += 1
    return bigrams


# given a list of unigrams, returns the list of trigrams present per window
def get_trigrams(l, window_size):
    list_trigrams = set()
    i = 0
    while i < len(l):
        j = i + 2
        temp = set()
        while j < min(i + window_size, len(l)):
            #if l[j] not in stopwords and l[j-2] not in stopwords:
            if l[j-2] + "_" + l[j-1] + "_" + l[j] in vocab:
                temp.add(l[j-2] + "_" + l[j-1] + "_" + l[j])
            j += 1
        list_trigrams.update(temp)
        i += 1
    return list(list_trigrams)


# given a list of unigrams, returns the list of unigrams present per window
def get_unigrams(l, window_size):
    list_unigrams = set()
    i = 0
    while i < len(l):
        temp = set()
        j = i
        while j < min(i + window_size, len(l)):
            #if l[j] not in stopwords:
            if l[j] in vocab:
                temp.add(l[j])
            j += 1
        list_unigrams.update(temp)
        i += 1
    return list(list_unigrams)


# A method to find (and return) entities from a piece of text,
# In turn, it updates the co-occurence matrix too.
def get_entities(s, window_size):
    if len(stopwords) == 0:
        load_stopwords()
    #s = clean(s.lower())

    list_unigrams = get_unigrams(s.split(), window_size); #print 'Unigrams = %d' %(len(list_unigrams))
    update(list_unigrams)

    #print s
    list_bigrams = get_bigrams(s.split(), window_size); #print 'Bigrams = %d' %(len(list_bigrams))
    update(list_bigrams)

    list_skipgrams = get_skipgrams(s.split(), window_size); #print 'Skipgrams = %d' %(len(list_skipgrams))
    update(list_skipgrams)

    list_trigrams = get_trigrams(s.split(), window_size) ; #print 'Trigrams = %d' %(len(list_trigrams))
    update(list_trigrams)

    '''
    till now, the each list has contributed to the co-occurence independently
    rather within each window, the co-ocurrence of bigrams and trigrams should
    also increase. Hence the following method
    '''
    updateCount = update_mixture(list_bigrams, list_skipgrams, list_unigrams, list_trigrams)
    #if updateCount > 0:
        #print 'Mixture co-occuraces found %d'%(updateCount)
    return list_unigrams, list_bigrams, list_skipgrams, list_trigrams


# updates the list of list of entities in the co-occurence matrix
def update(entities):
    global co
    #updateCount, selfCount = 0, 0
    #for window in entities:
    threadLock.acquire()
    for i in entities:
        for j in entities:
            #if i == j :
                #selfCount = selfCount + 1
                #print i,j
            co[int(vocab[i]), int(vocab[j]) ] += 1.0
            #updateCount += 1
    #if updateCount > 0:
        #print 'co-occuraces = %d. Freq = %d'%(updateCount, selfCount) #return updateCount
    threadLock.release()
    return None

# updates the co-occurence of entity i and j
def update_ij(i, j):
    global co
    updateCount = 0
    #i = i.replace(' ', '_')
    #j = j.replace(' ', '_')
    #if i in vocab and j in vocab:
    #print vocab[i], vocab[j] 
    co[int(vocab[i]), int(vocab[j]) ] += 1.0
    updateCount += 1
    #else :
    #    print 'No vocab entry for ',i,j
    
    return updateCount


# given a list of list of unigrams, bigrams, skipgrams, trigrams
# updates the co-occurece of each unigram-bigram-skipgram-trigram per each
# window
def update_mixture(l1, l2, l3, l4):
    G = []
    if len(l1) > 0 :
        G.extend(l1)
    if len(l2) > 0 :
        G.extend(l2)
    if len(l3) > 0 :
        G.extend(l3)
    if len(l4) > 0 :
        G.extend(l4)
    #Get unique elements
    G=set(G)
    G=list(G)
    #print 'Entities found',G
    updateCount = 0

    #for i in range(len(l1)):
    #l1.extend(l2).extend(l3).extend(l4)
    threadLock.acquire()
    for i, j in itertools.combinations(G, 2):
        co[int(vocab[i]), int(vocab[j]) ] += 1.0 #update_ij(i,j)        
        updateCount += 1
    threadLock.release()
    '''
        for a in l1[i]:
            for b in l2[i]:
                for c in l3[i]:
                    for d in l4[i]:
                        for j, k in itertools.permutations([a, b, c, d], 2):
                            update_ij(j, k)
    '''
    return updateCount


def load_stopwords():
    global stopwords

    f = open(stopwords_file, 'r')
    for i in f:
        stopwords.add(i.split()[0])
    return None


def setup_base(basefile):
    global co
    co = io.mmread(basefile).tolil()
    print 'loaded ', basefile, ' with ', co.shape
    return None

def mergeCo(file1, file2):
    global co
    co = io.mmread(file1).tolil()
    print 'loaded ', file1, ' with ', co.shape ,' in ', start_time - time.time() ,' time.'
    co2 = io.mmread(file2).tolil()
    print 'loaded ', file2, ' with ', co2.shape ,' in ', start_time - time.time(), ' time.'

    #for i in range(co.shape[0] - 1):
    #    for j in range(co.shape[1] - 1):
    #        if j in co2.rows[i]:
    #            co[i,j] += co2[i,j]
    #c1 = co.tocoo()
    c2 = co2.tocoo()
    co2 = co2.tolil()
    updateCount = 0 
    for i,j,v in itertools.izip(c2.row, c2.col, c2.data):
        co[i,j] += co2[i,j]
        updateCount +=1
        if updateCount % 100000 == 0:
            print 'Updated %d in CO' %(updateCount)


def populate(filename):
    count = 1
    f = open(filename, 'r')
    for i in f:
        if count % 10000 == 0:
            print count, 'number of lines done for', filename
        get_entities(i, 10)
        count += 1
    return None


def main():
    global stem_cache, vocab
    if len(sys.argv) < 5:
        print 'Usage : python2.7 pmi_index.py base_co.npy/None \
            vocab.pickle output_file inputFolder'

        print ' Example : python2.7 pmi_index.py ./co-occuranceFiles/initial_co.mtx \
        vocab.pickle ./co-occuranceFiles/one_co "/scratch/home/prince/mall-main/src/users/manjunath/entice/Priya/Sentences"'
        #print 'Usage : python2.7 pmi_index.py base_co.npy/None \
        #    vocab.pickle output_file file1 file2 ....'
        exit()

    basefile = sys.argv[1]
    if basefile != 'None':
        setup_base(basefile)

    vocab = pickle.load(open(sys.argv[2], 'r'))
    output_file = sys.argv[3]

    #files = sys.argv[4:]  # files to populate
    inputFolder = sys.argv[4]


    '''
    populate material
    '''
    counter = 0
    threadCount = 0
    threads = []
    start_time = time.time()
    for f in os.listdir(inputFolder):
        counter += 1
        print 'File : ', f        
        #populate(join(inputFolder,f)) #populate(f)
        #end_time = time.time()
        #print 'completed ',str(counter),'files. Processed but not dumped', f, 'in', end_time - start_time, 'time'


        if threadCount < maxThreads :
            try:
                #print 'processing Query id %s, cand %d'%(qId, candCount)
                thread1 = fileThread(threadCount, join(inputFolder,f), counter)
                thread1.start()
                threads.append(thread1)
            except:
                print "Error: unable to start thread for %d" %threadCount
            threadCount = threadCount + 1

        if threadCount==maxThreads: 
            # Wait for all threads to complete
            for t in threads:
                t.join()
            #print ' %d Threads ended. This tooks %0.2f minutes'%(maxThreads,((datetime.now()-start_time).total_seconds())/60)
            threadCount = 0
        

        if counter%100 == 0 :
            io.mmwrite(output_file + str(counter), co)
            end_time = time.time()
            print 'Completed and dumped', str(counter), ' files in', end_time - start_time, 'time'

    io.mmwrite(output_file, co)
    end_time = time.time()
    print 'completed and dumped ', f, 'in ', end_time - start_time, ' time'
    return None


#if __name__ == '__main__':
#    main()


#testing
#vocab = pickle.load(open('vocab.pickle', 'r'))

#vocab.csv has word to num_of_occurance 
'''
highestIndex = 0
with open('vocab.csv', 'r') as fr:
    for line in fr:
        line = line.strip()
        index = line.split(',')[-1]
        word = line.replace(index,'') #cutoff the index
        word = word.replace(word[len(word)-1] , '') #cut off the last ','
        vocab[word] = index
        if int(index)> highestIndex:
            highestIndex = int(index)
print 'Loaded vocab of size %d '%(len(vocab))

co = lil_matrix((highestIndex, highestIndex) ) #co = lil_matrix((len(vocab), len(vocab)) ) #co = np.zeros((len(vocab), len(vocab)))
#if 'Barack_Obama' in vocab:     print 'Ha'
sentence1 = 'Barack Obama is the 44th and current President of the United States. He is the first African American to hold the office and the first president born outside the continental United States. Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School,'
sentence2 = 'Michelle Obama (born January 17, 1964) is an American lawyer, writer, and First Lady of the United States. She is married to the 44th and current President of the United States, Barack Obama, and is the first African-American First Lady. Raised on the South Side of Chicago, Illinois, Obama is a graduate of Princeton University and Harvard Law School, and spent her early legal career working at the law firm Sidley Austin, where she met her husband'
'''
'''
myList = get_unigrams(sentence1.split(), cooccurence_window)
print myList
update(myList)
myList = get_bigrams(sentence1.split(), cooccurence_window); 
print myList
update(myList)
'''
'''
get_entities(sentence1, cooccurence_window);
get_entities(sentence2, cooccurence_window);
io.mmwrite('initial_co', co)
print 'Wrote CO with ', co.shape
pickle.dump(vocab, open('vocab.pickle','wb'))
print 'Wrote vocab with ',len(vocab)
'''
'''
update(myList); #print 'co-occuraces found %d' %(update(myList))
myList = get_bigrams(sentence2.split(), cooccurence_window);
update(myList); #print 'co-occuraces found %d' %(update(myList))
'''


#merge CO files
if len(sys.argv) < 3:
    print 'Usage error'
    exit()
file1 = sys.argv[1]
file2 = sys.argv[2]
output_file = sys.argv[3]
mergeCo(file1, file2)
io.mmwrite(output_file, co)
'''
#CO statistics
if len(sys.argv) < 1:
    print 'Usage error'
    exit()
basefile = sys.argv[1]
setup_base(basefile)
#number of non_zero rows
row_count = 0
row_sum = 0
for i in range(co.shape[0]):
    #find row average non zero
    if co[i].getnnz() > 0:
        row_sum += co[i].getnnz()
        row_count += 1
    if i%1000 == 0:
        print 'row_count = %d, row_sum = %d, row_avg = %0.4f' %(row_count, row_sum, 1.0 * row_sum/row_count)
print 'FINAL : row_count = %d, row_sum = %d, row_avg = %0.4f' %(row_count, row_sum, 1.0 * row_sum/row_count)
'''



