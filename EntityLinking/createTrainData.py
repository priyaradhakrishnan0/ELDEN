# Create train and test data for link classissifier from CONNL/AIDA dataset and TAC dataset,
# using the candidates generated from Pershina et al (for AIDA) and Wikipedia dump (for TAC).
import math, datetime
import os, re
from os import listdir, fsync, sys
from os.path import isfile, join
import re
from pymongo import MongoClient
from collections import OrderedDict
import requests
from requests.exceptions import ConnectionError
import threading
import Queue
from datetime import datetime
import cPickle as pickle

#Global variables set by user
datasetFlag, trainflag, restartFromQid = False, False, 0
#DATABASE SETUP
dbIP = '10.16.32.103' #db on momo 104' #db on dosa 
dbPort = 27017
client = MongoClient(dbIP, dbPort)
#client = MongoClient(); #db on localhost 
LinkDb = client.wikilinks
AnchorDb = client.anchorDB
wikiPageDb = client.wikiPageDB

#Directory for entities and candidates source
#sourceCandDir ='/media/New Volume/Datasets/'#PPRforNED-master/AIDA_candidates' #lenovo laptop
sourceCandDir ='/scratch/home/priya/' #dosa
#AIDA
AIDAsourceCandTest =sourceCandDir+'AIDAforNED/PART_1001_1393'
AIDAsourceCandTrain =sourceCandDir+'AIDAforNED/PART_1_1000'
#TAC
TACsourceCandTest =sourceCandDir+'TACforNED/Test.csv'
TACsourceCandTrain =sourceCandDir+'TACforNED/Train.csv'
EDqueriesFile = sourceCandDir+'EDqueries.csv'#File with entity pairs for which ED is to be calculated

#API url
ED_WLM_server_address = 'http://10.16.32.104:1339/ED/' #'http://10.16.32.104:1337/ED/' # #on dosa and vagrant ('http://10.16.32.105:1337/ED/') with 'eval3'  weights
ED_PMI_server_address = 'http://10.16.32.104:1338/ED/'#'http://10.16.34.115:1338/ED/' #'http://10.16.32.103:1338/ED/' # #'http://10.16.32.103:1338/ED/' #on mallGPU and momo with 'eval5.2' weights #'http://10.16.32.104:1337/ED/' #with 'eval5.1' weights #'http://10.16.34.115:1338/ED/' with 'eval3' weights  #'http://10.16.34.115:1337/ED/' with 'eval' weights
s = requests.Session()
a = requests.adapters.HTTPAdapter(max_retries=3)
s.mount('http://', a)
EDServerInError = False #When server is at fault set this true, to hault the program
PMI_server_address = 'http://10.16.32.104:2337/pmi/list?node=' #PMI list server defaults to port 2337
PMIServerInError = False #When server is at fault set this true, to hault the program

vocab_file = "vocab.pickle"
vocab, inv_vocab = {}, {}

#Thread Variables
maxThreads = 5 #max threads running at a time
threadLock = threading.Lock()
threadCount = 0
threadLock = threading.Lock()
start_time = datetime.now()

#Global variables modified by threads
candCount, priProb, linkFlag, entWikiName, multiQueryDocs, entityMention = 0, {}, 0, '', [], ''
pageCandPro = {} #max prior probability value of cand across mentions in this document
mentionCandList, mentionContext, Flist, candList, context, edQueuries = {}, {}, {}, [], '', []
maxPriProb , mentionMap, mentionPbMap, assignments, assignmentPMIlists, candidatePMIlists , entityMention, qId = 0.0, {}, {}, {}, {}, {}, '', ''

class myThread (threading.Thread):
    def __init__(self, threadID, fp, datasetFlag, trainFlag):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.fp = fp
        self.datasetFlag = datasetFlag
        self.trainFlag = trainFlag
    def run(self):
        print "Starting thread : " + str(self.threadID)+" on "+str(self.fp.name)
        processFile(self.fp, self.datasetFlag, self.trainFlag)
        self.fp.close()
        print "Exiting thread : " + str(self.threadID)

class candidateThread (threading.Thread):
    def __init__(self, threadID, line, candCount):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.line = line
        self.candCount = candCount
    def run(self):
        #print "Starting candidate thread : " + str(self.threadID)
        processCandidate(self.line, self.candCount)
        #print "Exiting candidate thread : " + str(self.threadID)

#HYPER PARAMERS
WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
contextSize = 15 #Number of words to left and right of the mention, considered as context.
CollationMethod = False #True for TAGME method of finding REL from rho. False for Yamada's 2-step method of finding rho(candidate).

#Evoke once during initialization
def loadVocab():
	global vocab, inv_vocab
	load_start_time = datetime.now()
	vocab = pickle.load(open(vocab_file, 'r'))
	inv_vocab = {v:k for k,v in vocab.iteritems()}
	print 'Vocab maps loaded in ', str(((datetime.now()-load_start_time).total_seconds())/60), 'minutes time'
	print 'PMIlistServer contacted %s' %PMI_server_address

#returns list of top K pmi entities with entry in W matrix
def getPMIentities(entityName):
	pmiEntities = []
	try:
		r_pmi = s.get(PMI_server_address+entityName, headers={'Connection':'close'})
		if r_pmi.status_code == 200:
			pmiEntities = [inv_vocab[k] for k in r_pmi.json() ]
			#to restrict to wikiEntities 
			#return [k for k in pmiEntities if getPageId(k) != 0]
			#return pmiEntities 
	except ConnectionError as e: 
		print  e
		PMIServerInError = True
	return pmiEntities 

#Get link-probability (Probability that an occurrence of a is an anchor pointing to some Wikipedia page) of this 'anchor'
def lp(anchor):
	lp = 0
	docs = AnchorDb.anchors.find({'anchor':anchor.lower()},{'total_freq':1, 'anchor_freq':1, '_id':0}) #print docs.count()
	for doc in docs:
		if doc['total_freq'] + doc['anchor_freq'] > 0:
			lp = 1.0 * doc['anchor_freq'] / (doc['total_freq'] + doc['anchor_freq'])
			break;
	return lp;

#Get unique elements of a list. Proved in https://www.peterbe.com/plog/uniqifiers-benchmark
def getUnique(seq): 
	newMap = {}
	for e in seq:
		newMap[e] = 1
	return newMap.keys()

def getInlinks (pagename):
	inlinks = []
	linksReturned = LinkDb.inlinks.find({'page':pagename},{'link':1, '_id':0})
	for doc in linksReturned:
		inlinks.extend(doc['link'])
	inlinks = getUnique(inlinks)
	return inlinks

def getOutlinks (pagename):
	outlinks = []
	linksReturned = LinkDb.outlinks.find({'page':pagename},{'link':1, '_id':0})
	for doc in linksReturned:
		outlinks.extend(doc['link'])
	outlinks = getUnique(outlinks)
	return outlinks

#WLM calculated using outlinks of E1 and E2 
def getWLMout (entity1, entity2):
	linkReturned = LinkDb.WLM.find_one({'E1':entity1, 'E2':entity2}, {'WLMout':1, '_id':0})
	if linkReturned == None : #2nd try wth entities inversed
		linkReturned = LinkDb.WLM.find_one({'E1':entity2, 'E2':entity1}, {'WLMout':1, '_id':0})	#print 'WLMout'
	if linkReturned != None: #print linkReturned['WLMout']
		return linkReturned['WLMout']
	else:#print '-1.0'
		return -1.0 #i.e return Nones
	
#WLM calculated using inlinks to E1 and E2 
def getWLMin (entity1, entity2):
	linkReturned = LinkDb.WLM.find_one({'E1':entity1, 'E2':entity2}, {'WLMin':1, '_id':0})
	if linkReturned == None: #2nd try wth entities inversed
		linkReturned =  LinkDb.WLM.find_one({'E1':entity2, 'E2':entity1}, {'WLMin':1, '_id':0})	#print 'WLMin'
	if linkReturned != None: #print linkReturned['WLMin']
		return linkReturned['WLMin']
	else: #print '-1.0'
		return -1.0 #no record found
	
#Get wikipedia pageId of this title
def getPageId(title):
	docs = wikiPageDb.wikiPageTitle.find({'title':title}, {'pageId':1,'_id':0})
	if docs.count() > 0 :
		return docs[0]['pageId']
	else :
		return 0

#Get wikipedia title of this pageId
def getTitle(pageId):
	docs = wikiPageDb.wikiPageTitle.find({'pageId':pageId}, {'title':1,'_id':0});#print 'Title = %s' %docs[0]['title']
	return docs[0]['title']

#Get total anchors count 
def getEntityPrior (pgId):
	docC = AnchorDb.anchors.find({'pages.page_id':pgId}).count()	
	#total = 8778720 #AnchorDb.anchors.count() #total number of anchors
	#print 'Page %d has %d anchors linking to it. Prior Prob %0.8f'%(pgId, docC, 1.0 * docC / 8778720)
	return  (1.0 * docC / 8778720);

#Get link prior- probabilities of pages that mention the string 'anchor'
def getPageProbability (anchor):
	pageProb = {} #	page : Prob(page)
	docs = AnchorDb.anchors.find({'anchor':anchor.lower()},{'pages':1, 'anchor_freq':1, '_id':0}) 
	total_pages = 0 #total number of pages that link to this anchor
	for doc in docs:
		for page in doc['pages']: #	print page['page_id']
			pageProb[page['page_id']] = 1.0*page['page_freq']/doc['anchor_freq']
			#print str(page['page_id'])+' === ' + str(1.0*page['page_freq']/doc['anchor_freq'])
	return pageProb;

def rel(anchor1, anchor2):
	SR = 0.0;
	inlinks1 = getInlinks(anchor1); print anchor1 + ' has inlinks '+ str(len(inlinks1)) # print anchor1 + ' has unique inlinks '+ str(len(inlinks1))
	inlinks2 = getInlinks(anchor2) ; print anchor2 + ' has inlinks '+ str(len(inlinks2)) # print anchor2 + ' has unique inlinks '+ str(len(inlinks2))
	commonInlinks = set(inlinks1) & set(inlinks2)
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonInlinks) > 0 :
		num = math.log10(max(len(inlinks1), len(inlinks2))) - math.log10(len(commonInlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(inlinks1), len(inlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den )
	#print 'rel = %0.6f' %SR
	return SR;

#inlinks are densified (increased) using PMI entities
def relDense(anchor1, anchor2):
	SR = 0.0;
	inlinks1 = getInlinks(anchor1); print anchor1 + ' has inlinks '+ str(len(inlinks1))
	pmiEntities = getPMIentities(anchor1)
	if len(pmiEntities) > 0:
		inlinks1.extend(pmiEntities); print anchor1 + ' has inlinks is now '+ str(len(inlinks1))
		inlinks1 = getUnique(inlinks1) ; print anchor1 + ' has densified inlinks '+ str(len(inlinks1))

	inlinks2 = getInlinks(anchor2) ; print anchor2 + ' has inlinks '+ str(len(inlinks2))
	pmiEntities = getPMIentities(anchor2)
	if len(pmiEntities) > 0:
		inlinks2.extend( pmiEntities)
		inlinks2 = getUnique(inlinks2); print anchor2 + ' has densified inlinks '+ str(len(inlinks2))
	commonInlinks = set(inlinks1) & set(inlinks2)
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonInlinks) > 0 :
		num = math.log10(max(len(inlinks1), len(inlinks2))) - math.log10(len(commonInlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(inlinks1), len(inlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den )
	print 'relDense = %0.6f' %SR
	return SR;

def relReverse(anchor1, anchor2):#WLM using OutLinks (i.e reverse of Inlinks)
	SR = 0.0;
	outlinks1 = getOutlinks(anchor1); #print anchor1 + ' has outlinks '+ str(len(outlinks1))
	outlinks2 = getOutlinks(anchor2) ; #print anchor2 + ' has outlinks '+ str(len(outlinks2))
	commonOutlinks = set(outlinks1) & set(outlinks2)
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonOutlinks) > 0 :
		num = math.log10(max(len(outlinks1), len(outlinks2))) - math.log10(len(commonOutlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(outlinks1), len(outlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den )
	#print 'relRev = %0.6f' %SR
	return SR;

#inlinks are densified (increased) using PMI entities
def relDense2(anchorlinks, anchor1, anchor2):
	SR = 0.0;
	#inlinks1 = getInlinks(anchor1); print anchor1 + ' has inlinks '+ str(len(inlinks1))
	pmiEntities = getPMIentities(anchor1);
	if len(pmiEntities) > 0 :
		anchorlinks.extend(pmiEntities); #print anchor1 + ' has inlinks is now '+ str(len(inlinks1))
		anchorlinks = getUnique(anchorlinks) ; #print anchor1 + ' has densified inlinks '+ str(len(inlinks1))

	inlinks2 = getInlinks(anchor2) ; #print anchor2 + ' has inlinks '+ str(len(inlinks2))
	pmiEntities = getPMIentities(anchor2)
	if len(pmiEntities) > 0:
		inlinks2.extend( pmiEntities)
		inlinks2 = getUnique(inlinks2); #print anchor2 + ' has densified inlinks '+ str(len(inlinks2))
	commonInlinks = set(anchorlinks) & set(inlinks2)
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonInlinks) > 0 :
		num = math.log10(max(len(anchorlinks), len(inlinks2))) - math.log10(len(commonInlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(anchorlinks), len(inlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den )
	#print 'relDense = %0.6f' %SR
	return SR;

#relDense3(candInlinks, candWikiName, assgWikiTitle, assgEntity);
def relDense3(anchorlinks, anchor1, anchor2, anchor2Id):
	global assignmentPMIlists, candidatePMIlists
	SR = 0.0;
	#inlinks1 = getInlinks(anchor1); print anchor1 + ' has inlinks '+ str(len(inlinks1))
	pmiEntities = candidatePMIlists[anchor1]
	if len(pmiEntities) > 0 :
		anchorlinks.extend(pmiEntities); #print anchor1 + ' has inlinks is now '+ str(len(inlinks1))
		anchorlinks = getUnique(anchorlinks) ; #print anchor1 + ' has densified inlinks '+ str(len(anchorlinks))

	inlinks2 = getInlinks(anchor2) ; #print anchor2 + ' has inlinks '+ str(len(inlinks2))
	pmiEntities = assignmentPMIlists[anchor2Id] 
	#pmiEntities = getPMIentities(anchor2)
	if len(pmiEntities) > 0:
		inlinks2.extend( pmiEntities)
		inlinks2 = getUnique(inlinks2); #print anchor2 + ' has densified inlinks '+ str(len(inlinks2))
	commonInlinks = set(anchorlinks) & set(inlinks2)
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonInlinks) > 0 :
		num = math.log10(max(len(anchorlinks), len(inlinks2))) - math.log10(len(commonInlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(anchorlinks), len(inlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den )
	#print 'relDense3 = %0.6f' %SR
	return SR;

def relReverse2(anchor1links, anchor2):#WLM using OutLinks (i.e reverse of Inlinks)
	SR = 0.0;
	#outlinks1 = getOutlinks(anchor1); print anchor1 + ' has outlinks '+ str(len(outlinks1))
	outlinks2 = getOutlinks(anchor2) ; #print anchor2 + ' has outlinks '+ str(len(outlinks2))
	commonOutlinks = set(anchor1links) & set(outlinks2); 
	#if type(len(commonOutlinks)) is int : print anchor2 + ', Common outlinks '+ str(len(commonOutlinks))
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonOutlinks) > 0 :
		num = math.log10(max(len(anchor1links), len(outlinks2))) - math.log10(len(commonOutlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(anchor1links), len(outlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den ); #print 'relRev2 = %0.6f' %SR
	return SR;

def rel2(anchor1links, anchor2):
	SR = 0.0;
	inlinks2 = getInlinks(anchor2) ; #print anchor2 + ' has inlinks '+ str(len(inlinks2))
	commonInlinks = set(anchor1links) & set(inlinks2) ; 
	#if type(len(commonInlinks)) is int : print anchor2 + ', Common inlinks '+ str(len(commonInlinks))
	#WikiPageCount = 5123051 #Number of articles in Wikipedia as of 9 Apr,2016
	if len(commonInlinks) > 0 :
		num = math.log10(max(len(anchor1links), len(inlinks2))) - math.log10(len(commonInlinks)); #print ' Numr = '+str(num)
		den = math.log10(WikiPageCount) - math.log10(min(len(anchor1links), len(inlinks2))); #print ' Denr = '+str(den)
		SR = 1 - ( num * 1.0 / den ); #print 'rel2 = %0.6f'%SR
	return SR;

#Get map of page-id to page_freq if the page mentions the string 'anchor'
def getPagesMap (anchor):
	pageMap = {}
	docs = AnchorDb.anchors.find({'anchor':anchor.lower()},{'pages':1, '_id':0}) #print docs.count()
	for doc in docs:
		for page in doc['pages']: #	print page['page_id']
			pageMap[page['page_id']] = page['page_freq']		
	return pageMap;

#Get entity mentions of a query string from given text
def mentions(text):
	mentions = {} #we retun only the keys of this dictionary.
	#text = text.lower();
	text = text.replace(',',' ').replace('.',' ').replace(';',' ').replace("\\s+", " ")
	splits = re.split('_|:| ', text)
	splits = [ k for k in splits if k.strip() != '']
	i = 0
	while i < len(splits)-1: #print 'start i = %d' %i
		currLP = 0.0
		for j in range(5,0,-1):
			if(i+j) > len(splits):
				continue;
			cMention = "";
			for k in range(i,(i+j),1):
				if k < len(splits):
					cMention += splits[k] + " "; #.lower().trim()
			
			cMention = cMention.strip();
			#check if cMention != ''
			lp_cMention = lp(cMention); #print 'i = %d, j = %d cMention :%s, lp : %0.6f' %(i,j,cMention, lp_cMention)
			if lp_cMention > 0.01 and lp_cMention > currLP :
				mentions[cMention] = lp_cMention;
				currLP = lp_cMention; #	print 'Storing at i = %d, j = %d, mention = %s, lp = %0.6f' %(i,j, cMention, lp_cMention)
				i = i + j - 1;
				break;
		i += 1;
	
	#print ' Before Pruning : '
	#for mention in mentions:
	#	print '%s : %0.6f' %(mention, mentions[mention])
	#print mentions
	mentions_sort = OrderedDict(sorted(mentions.items(), key=lambda t: t[1], reverse=True))
	#print mentions_sort
	'''
	mentionsList = mentions_sort.keys();
	print '#mentions = %d' %len(mentionsList)
	print mentionsList
	mentionsList.append('karnataka')
		
	curr = "";
	if len(mentionsList) > 0 :
		curr = mentionsList[0]
		mentionsList.remove(curr) # compare all other than current
		i = 0
		while  i < (len(mentionsList)):
			print 'curr = %s, i = %d, mentionsList[i] = %s ' %(curr, i, mentionsList[i]) 
			if curr in mentionsList[i]:
				print 'curr %0.6f i %0.6f ' %(mentions[curr], mentions[mentionsList[i]]) 
				if mentions[curr] >= mentions[mentionsList[i]]:
					#pruneMentions.add(i);
					mentions.remove(mentionsList[i]);
					i = i-1
				else :
					mentions.remove(curr);
					curr = mentionsList[i]
			else:
				curr = mentionsList[i]
			i += 1
	
	print ' After Pruning : '
	print mentionsList
	#for mention in mentions:
	#	print '%s : %0.6f' %(mention, mentions[mention])
	'''
	return mentions_sort;#return mentions_sort.keys();
	
def openOutputFile(datasetFlag, trainflag):
	if(datasetFlag): #AIDA dataset
		if (trainflag):
			fOut = open('AIDA_candidates_train.csv_ED7_0','a') #Opening in append mode as multiple calls per AIDA dataset run. Please flush out file manually before start of run
		else:
			fOut = open('AIDA_candidates_test.csv_ED7_0','a') 
	else : #TAC dataset
		if (trainflag):
			fOut = open('TAC_candidates_train.csv_ED7_0','a') #Opening in append mode as called once per document in TAC dataset run. Please flush out file manually before start of run
		else:
			fOut = open('TAC_candidates_test.csv_ED7_0','a') 
	return fOut;

def conciseContext(mention, context, contextSize):
	chunk = []
	if len(mention.split()) == 1:
		if mention in context.split() :
		    p = context.split().index(mention)
		    #print 'p = %d'%p
		    if (p - contextSize) > 0:
		       lb = p - contextSize
		    else :
		    	lb = 0
		    if (p + contextSize) < len(context.split()):
		       rb = p + contextSize
		    else:
		    	rb = len(context.split())
		    #print 'left boudary %d, right boundary %d' %(lb,rb)
		    chunk = context.split()[lb:rb]
		    #print chunk	    	
	elif len(mention.split()) > 1:
		mention_substring = mention.split(); #print 'Mention has %d parts'%len(mention_substring)
		p  = 0
		for word in context.split():			
			if word==mention_substring[0]:
				if (p + 1) < len(context.split()):
					#print 'p = %d in %d'%(p, len(context.split()))
					if context.split()[p+1] == mention_substring[1]:					
						if (p - contextSize) > 0:
							lb = p - contextSize
						else :
							lb = 0
						if (p + contextSize) < len(context.split()):
							rb = p + contextSize
						else:
							rb = len(context.split())
						#print 'left boudary %d, right boundary %d' %(lb,rb)
						chunk = context.split()[lb:rb]
						#print chunk;
						break;
			p = p + 1 #position of word

	return ' '.join(chunk)

#print the features of file-pointer fp as train(true) or test(false)
def processFile(fp, datasetFlag, trainflag):
	global candCount
	global priProb, linkFlag, entWikiName, multiQueryDocs 
	if not datasetFlag : #TAC		
		if (trainflag):
			#fOut = open('TAC_candidates_train.csv_current_EA','w')#open('TAC_candidates_train.csv_current'+sys.argv[3],'w')
			multiQueryDocs = ['APW_ENG_20080406.0055.LDC2009T13', 'AFP_ENG_20050917.0434.LDC2007T07', 'AFP_ENG_20070718.0412.LDC2009T13', 'AFP_ENG_20070219.0141.LDC2009T13', 'APW_ENG_20070814.1496.LDC2009T13', 'AFP_ENG_20030915.0304.LDC2007T07', 'NYT_ENG_20070127.0051.LDC2009T13', 'AFP_ENG_20081016.0103.LDC2009T13', 'CNA_ENG_20070719.0042.LDC2009T13', 'AFP_ENG_20070906.0251.LDC2009T13', 'AFP_ENG_20021027.0287.LDC2007T07', 'CNA_ENG_20070122.0041.LDC2009T13', 'APW_ENG_20050916.0551.LDC2007T07', 'NYT_ENG_20070820.0154.LDC2009T13', 'AFP_ENG_20070303.0399.LDC2009T13', 'APW_ENG_20080318.1951.LDC2009T13', 'AFP_ENG_20070420.0530.LDC2009T13', 'LTW_ENG_20070804.0027.LDC2009T13', 'LTW_ENG_20081216.0003.LDC2009T13', 'AFP_ENG_20020617.0564.LDC2007T07', 'XIN_ENG_20070310.0142.LDC2009T13', 'AFP_ENG_20080730.0702.LDC2009T13', 'AFP_ENG_20071005.0680.LDC2009T13', 'AFP_ENG_19940616.0276.LDC2007T07', 'AFP_ENG_20030724.0396.LDC2007T07', 'AFP_ENG_20060518.0628.LDC2007T07', 'AFP_ENG_20081214.0187.LDC2009T13', 'AFP_ENG_20070328.0183.LDC2009T13', 'CNA_ENG_20070530.0010.LDC2009T13', 'AFP_ENG_20070218.0109.LDC2009T13', 'APW_ENG_20070804.0803.LDC2009T13', 'NYT_ENG_20070429.0117.LDC2009T13', 'LTW_ENG_20080524.0040.LDC2009T13', 'APW_ENG_20071201.0247.LDC2009T13', 'CNA_ENG_20061204.0009.LDC2007T07', 'LTW_ENG_19950511.0116.LDC2007T07', 'CNA_ENG_20070926.0026.LDC2009T13', 'CNA_ENG_20080725.0042.LDC2009T13', 'NYT_ENG_19980102.0443.LDC2007T07', 'APW_ENG_20070522.0220.LDC2009T13', 'CNA_ENG_20070107.0018.LDC2009T13', 'NYT_ENG_20071118.0132.LDC2009T13', 'AFP_ENG_20070104.0236.LDC2009T13', 'AFP_ENG_20070112.0437.LDC2009T13', 'APW_ENG_20070420.0224.LDC2009T13', 'NYT_ENG_20080904.0104.LDC2009T13', 'AFP_ENG_19960206.0102.LDC2007T07', 'AFP_ENG_20070209.0112.LDC2009T13', 'NYT_ENG_20080418.0024.LDC2009T13', 'AFP_ENG_20070131.0686.LDC2009T13', 'APW_ENG_20071130.1274.LDC2009T13', 'AFP_ENG_20061108.0665.LDC2007T07', 'AFP_ENG_20050914.0157.LDC2007T07', 'CNA_ENG_20070906.0043.LDC2009T13', 'NYT_ENG_20070119.0135.LDC2009T13', 'AFP_ENG_20070823.0152.LDC2009T13', 'APW_ENG_20080725.0086.LDC2009T13', 'APW_ENG_20070109.0131.LDC2009T13', 'AFP_ENG_20070307.0294.LDC2009T13', 'APW_ENG_20010502.1067.LDC2007T07', 'AFP_ENG_20051122.0140.LDC2007T07', 'XIN_ENG_20071009.0060.LDC2009T13', 'AFP_ENG_20061108.0685.LDC2007T07', 'AFP_ENG_20081031.0637.LDC2009T13', 'APW_ENG_20070821.0708.LDC2009T13', 'CNA_ENG_20070804.0015.LDC2009T13', 'APW_ENG_20081120.0295.LDC2009T13', 'NYT_ENG_20071231.0021.LDC2009T13', 'APW_ENG_20070304.0061.LDC2009T13', 'AFP_ENG_19950512.0168.LDC2007T07', 'CNA_ENG_20071005.0039.LDC2009T13', 'AFP_ENG_20070327.0316.LDC2009T13', 'LTW_ENG_20070724.0114.LDC2009T13', 'AFP_ENG_20060907.0454.LDC2007T07', 'AFP_ENG_20050918.0351.LDC2007T07', 'AFP_ENG_20081016.0047.LDC2009T13', 'AFP_ENG_20030525.0468.LDC2007T07', 'AFP_ENG_20031129.0157.LDC2007T07', 'NYT_ENG_20080507.0269.LDC2009T13', 'AFP_ENG_19960413.0028.LDC2007T07']
		else:
			#fOut = open('TAC_candidates_test.csv_current_EA','w') #open('TAC_candidates_test.csv_current'+sys.argv[3],'w')
			multiQueryDocs = ['eng-NG-31-142146-10009724', 'eng-WL-11-174595-12967698', 'eng-NG-31-141971-9990370', 'APW_ENG_20080309.0622.LDC2009T13', 'APW_ENG_20071210.0471.LDC2009T13', 'eng-NG-31-142149-10023737']
	global pageCandPro, mentionCandList, mentionContext, Flist, candList, context, edQueuries 
	global maxPriProb , mentionMap, mentionPbMap, assignments, assignmentPMIlists, candidatePMIlists, entityMention, qId, EDServerInError, restartFromQid, PMIServerInError

	threadCount = 0
	threads = []
	
	for line in fp.readlines():
		if str.startswith(line, "ENTITY"):
			if datasetFlag : #AIDA
				entityMention = line[12:line.find('normalName:')].strip()
				qId = os.path.basename(fp.name)
			else : #TAC
				entityMention = line[12:line.find('qid:')].strip()
				qId = line[line.find('qid:')+4:line.find('docId:')].strip() 
				docId = line[line.find('docId:')+6:line.find('url:')].strip()  
				if restartFromQid != 0:
					if qId != restartFromQid:
						continue
					elif qId == restartFromQid:
						restartFromQid = 0
						print 'processing from Query id %s'%qId
			
			priProb = getPageProbability(entityMention)
			if len(priProb)> 0:
				maxPriProb = max(priProb.values())
			else:
				maxPriProb = 0
			mentionMap = {} 
			mentionPbMap = {}

			if entWikiName != "": #This is next query. So print-out the previous query 
				if len(candList) > 0:
					mentionCandList[entWikiName] = candList
					if not datasetFlag : # TAC
						if docId in multiQueryDocs : #wait for remaining queries
							if len(mentionCandList)>1: 
								# Wait for all threads to complete before writing output
								for t in threads:
									t.join()
								writeFeatures(datasetFlag, trainflag, mentionCandList,Flist,pageCandPro,edQueuries)
								mentionCandList = {}
								Flist = {}
								pageCandPro ={}
								edQueuries = []								

						else :#single queried doc. So write-out the features
							# Wait for all threads to complete before writing output
							for t in threads:
								t.join()
							writeFeatures(datasetFlag, trainflag, mentionCandList,Flist,pageCandPro,edQueuries)
							mentionCandList = {}
							Flist = {}
							pageCandPro ={}
							edQueuries = []

				candidatePMIlists = {}
				entWikiName = ""
				linkFlag = 0 #False
				candList = []
				if context != '':
					mentionContext[entWikiName] = context
				context = ''
			
			if "url:http://en.wikipedia.org/wiki/" in line :
				entWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:].strip(); #print 'Entity wiki name : %s' %(entWikiName)

		elif str.startswith(line, "CONTEXT"):

			if not datasetFlag and restartFromQid != 0:
				if qId != restartFromQid:
					continue

			context = line.strip().replace("CONTEXT\t", "")
			context = conciseContext(entityMention, context, contextSize ) #contextSize=15 as in 15 words on either side of the mention.
			#print 'CONTEXT %s'%context			
			mentionMap = mentions(context); #mention - lp map #b
			#remove entityMention from mentionMap
			if entityMention in mentionMap:
				del mentionMap[entityMention]
			#print 'Possible mentions '; print (mentionMap)

			mentionPbMap = {} #print 'Mention : Number of pages '
			for mention in mentionMap:
				mentionPbMap[mention] = getPageProbability(mention) #print ' %s : %d' %(mention, len(mentionPbMap[mention]))
			#print '#PbMaps = %d'%(len(mentionPbMap))

			if not CollationMethod : #2-step method
				#Step 1 of 2-step method : Assign mention to Pb with max prior prob
				assignments = {}
				assignmentPMIlists = {}
				for mention in mentionPbMap: #PbMap = mentionPbMap[mention]
					Pb_max=max(mentionPbMap[mention].values()) 
					#print ' mention %s Pb_max %0.6f ' %(mention, Pb_max)
					if Pb_max > 0.9 :
						assignments[mentionPbMap[mention].keys()[mentionPbMap[mention].values().index(Pb_max)]] = Pb_max # assigning the key with the max value						
				#print 'Step 1 #assignments : %d ' %(len(assignments))
				#print assignments 
				if len(assignments)>0:
					for assgEntity in assignments:
						assignmentPMIlists[assgEntity] = getPMIentities(getTitle(assgEntity))
					#print 'Yamada method: Step 1 #assignments : %d , fetched PMI for %d' %(len(assignments), len(assignmentPMIlists))

		elif str.startswith(line, "CANDIDATE"):

			if not datasetFlag and restartFromQid != 0:
				if qId != restartFromQid:
					continue

			#populate candidate's PMI list for the threads to process
			if datasetFlag : #AIDA
				candWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:line.find('name:')].strip()
			else : #TAC
				candWikiNameMatch = line.rfind('/');#candWikiNameMatch = re.match('(.*)url:http://en.wikipedia.org/wiki/([^\s]+)', line)
				candWikiName = line[candWikiNameMatch+1:].strip()
			candidatePMIlists[candWikiName] = getPMIentities(candWikiName)

			if threadCount < maxThreads :
				try:
					#print 'processing Query id %s, cand %d'%(qId, candCount)
					thread1 = candidateThread(threadCount, line, candCount)
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
			
			candCount = candCount + 1
		
		if EDServerInError or PMIServerInError :
			print "QUIT : EDServer or PMIlistServer is OOS: Processed till %s" %(qId)
			break;	

	# Wait for all candidate threads to complete before writing output
	for t in threads:
		t.join()
	#writeFeatures called for TAC and AIDA. For TAC writes the last Entity. For AIDA, writes the present query
	writeFeatures( datasetFlag, trainflag, mentionCandList, Flist, pageCandPro, edQueuries)


#process the input line for candidate to write features 
def processCandidate(line, candCount):
	global entWikiName, Flist, candList, priProb, pageCandPro, maxPriProb, entityMention, qId, mentionPbMap, edQueuries, assignments, assignmentPMIlists, candidatePMIlists, Flist, datasetFlag, trainflag, threadLock, EDServerInError, PMIServerInError
	textSimiFeat1, textSimiFeat2 = '0','0'
	start = datetime.now()
	#print 'time#1', print datetime.datetime.now() #note print syntax. The ',' instead of ';' makes print-out on the same line
	f_list = [] #list of output features
	FNUM = 0 #feature number

	if entWikiName != "":	
		if datasetFlag : #AIDA
			candWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:line.find('name:')].strip()
		else : #TAC
			candWikiNameMatch = line.rfind('/');#candWikiNameMatch = re.match('(.*)url:http://en.wikipedia.org/wiki/([^\s]+)', line)
			candWikiName = line[candWikiNameMatch+1:].strip()
		candId = getPageId(candWikiName)
		#print ' Entity : %s, Candidate : %s, id : %d' %(entWikiName, candWikiName, candId)				
		#print 'time#2',	print datetime.datetime.now()
							
		if candId > 0 :
			candList.append(candId)	
			#Write query and candidate details to feature list
			f_list.append('qId:'+qId)
			f_list.append('cand:'+str(candId))	
			#F_0 : LABEL : Link or not
			FNUM = 0
			if candWikiName == entWikiName :
				f_list.append(str(FNUM)+':1'); linkFlag = 1 #True + 1
			else :
				f_list.append(str(FNUM)+':0'); linkFlag = 0
			#print ' Entity %s, link %d' %(entWikiName, linkFlag)

			#BASE FEATURES
			#F_1 : ENTITY PRIOR #F_1
			FNUM = FNUM + 1
			ePrior = getEntityPrior(candId)
			if ePrior :
				f_list.append(str(FNUM)+':'+str(ePrior))
			else:
				f_list.append(str(FNUM)+':0')

			#F_2 : PRIOR PROBABILITY #F_2
			FNUM = FNUM + 1
			if  candId in priProb:
				if candId in pageCandPro:
					if pageCandPro[candId] < priProb[candId]:
						pageCandPro[candId] = priProb[candId]
				else :
					pageCandPro[candId] = priProb[candId]
				f_list.append(str(FNUM)+':'+str(priProb[candId]))
				#print ' Entity %s, priPro %0.4f, ePrior %0.8f' %(entWikiName, priProb[candId], ePrior)
			else :
				f_list.append(str(FNUM)+':0')	

			#maximum value of prior prob of this mention. Constant for all candidates of the mention
			FNUM = FNUM + 1
			f_list.append(str(FNUM)+':'+str(maxPriProb))  #F_3

			#STRING SIMILARITY FEATURES
			#the title of candidate entity exactly equals the surface of mention m
			if entityMention==candWikiName:
				textSimiFeat1 = '1'
			else:
				textSimiFeat1 = '0'
			#the title of candidate entity contains or starts or ends with the surface of mention m
			if entityMention in candWikiName or candWikiName.startswith(entityMention) or candWikiName.endswith(entityMention):
				textSimiFeat2 = '1'
			else:
				textSimiFeat2 = '0'
			FNUM = FNUM + 1
			f_list.append(str(FNUM)+':'+textSimiFeat1)	#F_4				
			FNUM = FNUM + 1
			f_list.append(str(FNUM)+':'+textSimiFeat2)	#F_5

			#print 'time#3'; print datetime.datetime.now()					
			#Entity - Entity similarity features
			#WLM with inlinks
			REL, REL_out, REL_dense, REL_ED_wlm, REL_ED_pmi = 0.0, 0.0, 0.0, 0.0, 0.0 #in TAGME method
			rho, rho_out, rho_dense, rho_ED_wlm, rho_ED_pmi = 0.0, 0.0, 0.0, 0.0, 0.0 #in 2-step method
			#ED with WLM
			ED_WLM = 0.0
			ED_PMI = 0.0
			if len(mentionPbMap) > 0:
				candInlinks = getInlinks(candWikiName)
				candOutlinks = getOutlinks(candWikiName)
				reusedRelCount = 0
				ED_wlm, ED_pmi = 0.0, 0.0
				if len(candInlinks) > 0 or len(candOutlinks)> 0 :
					if (CollationMethod): # TAGME method
						for mention in mentionPbMap:							
							#print ' PbMap '#print PbMap
							vote = 0.0; vote_out = 0.0; vote_dense = 0.0; vote_ED_wlm = 0.0; vote_ED_pmi = 0.0
							for Pb in mentionPbMap[mention]: #PbMap:
								PbWikiTitle = getTitle(Pb)
								#log ED query
								#edQueuries.append((getTitle(Pb)+'#'+candWikiName).encode('utf8'))
								WLM_in = -1.0 #getWLMin(Pb, candId)
								WLM_out = -1.0 #getWLMout(Pb, candId); #print 'candId %d Pb %d WLM_in %0.8f WLM_out %0.8f '%(candId, Pb, WLM_in, WLM_out)
								WLM_dense = -1.0 #change to getWLMdense function and one run
								try:
									r_wlm = s.get(ED_WLM_server_address+PbWikiTitle+':'+candWikiName, headers={'Connection':'close'})
									if r_wlm.status_code == 200 and 'nan' not in r_wlm.text:
										ED_wlm = float(r_wlm.text)
									r_pmi = s.get(ED_PMI_server_address+PbWikiTitle+':'+candWikiName, headers={'Connection':'close'})
									if r_pmi.status_code == 200 and 'nan' not in r_pmi.text:
										ED_pmi = float(r_pmi.text)
										#if ED_wlm != 0.0 :
											#print 'candId %d Pb %d ED_wlm %0.8f '%(candId, Pb, ED_wlm)
								except ConnectionError as e: 
									print  e
									EDServerInError = True
								'''
								if WLM_in == -1.0:
									WLM_in = rel2(candInlinks, PbWikiTitle)
									if WLM_out == -1.0:
										WLM_out = relReverse2(candOutlinks, PbWikiTitle)
									LinkDb.WLM.insert({'E1':Pb, 'E2':candId, 'WLMin':WLM_in, 'WLMout':WLM_out}); #print 'relReverse stored for %s = %0.8f' %(str(Pb)+'#'+str(candId), WLM_out)
								else :
									if WLM_out == -1.0:
										WLM_out = relReverse2(candOutlinks, PbWikiTitle)
										LinkDb.WLM.insert({'E1':Pb, 'E2':candId, 'WLMin':WLM_in, 'WLMout':WLM_out}); #print 'relReverse stored for %s = %0.8f' %(str(Pb)+'#'+str(candId), WLM_out)
									else :
										reusedRelCount = reusedRelCount + 1
								if WLM_dense == -1.0:
								'''
								WLM_in = rel2(candInlinks, PbWikiTitle)
								WLM_out = relReverse2(candOutlinks, PbWikiTitle)
								WLM_dense = relDense2(candInlinks, candWikiName, PbWikiTitle)
								LinkDb.WLM.insert({'E1':Pb, 'E2':candId, 'WLMin':WLM_in, 'WLMout':WLM_out, 'WLMdense':WLM_dense}); #print 'relReverse stored for %s = %0.8f' %(str(Pb)+'#'+str(candId), WLM_out)

								vote = vote + (WLM_in * mentionPbMap[mention][Pb]) # rel(Pa,Pb) * Prior-prob(Pb)
								vote_out = vote_out + (WLM_out * mentionPbMap[mention][Pb]) # rel(Pa,Pb) * Prior-prob(Pb)	
								vote_dense = vote_dense + (WLM_dense * mentionPbMap[mention][Pb]) # rel(Pa,Pb) * Prior-prob(Pb)
								vote_ED_wlm = vote_ED_wlm + (ED_wlm * mentionPbMap[mention][Pb])
								vote_ED_pmi = vote_ED_pmi + (ED_pmi * mentionPbMap[mention][Pb])
								#print 'candId %d Pb %d WLM_in %0.8f WLM_out %0.8f ED_wlm %0.8f '%(candId, Pb, WLM_in, WLM_out, ED_wlm)

							#print 'MENTION %s, VOTE %0.6f, VOTE_out %0.6f, ReusedRelCount %d' %(mention,vote,vote_out,reusedRelCount)
							REL = REL + vote
							REL_out = REL_out + vote_out
							REL_dense = REL_dense + vote_dense
							REL_ED_wlm = REL_ED_wlm + vote_ED_wlm
							REL_ED_pmi = REL_ED_pmi + vote_ED_pmi
						#print 'RELinlinks for candidte %s is %0.6f. RELoutlinks is %0.6f. REL_ED_wlm is %0.6f. ReusedRelCount %d' %(candWikiName, REL, REL_out, REL_ED_wlm, reusedRelCount)
					else : #Yamada's 2-step method
						if len(assignments) > 0 :
							#print 'begin ', len(assignments)
							for assgEntity in assignments:
								assgWikiTitle = getTitle(assgEntity)
								#log ED query
								#edQueuries.append((getTitle(assgEntity)+'#'+candWikiName).encode('utf8'))
								#rhoIn = getWLMin(assgEntity, candId)
								#rhoOut = getWLMout(assgEntity, candId)
								#Add rhoDense here
								try:
									r_wlm = s.get(ED_WLM_server_address+assgWikiTitle+':'+candWikiName, headers={'Connection':'close'})
									if r_wlm.status_code == 200 and 'nan' not in r_wlm.text:
										ED_wlm = float(r_wlm.text)
									r_pmi = s.get(ED_PMI_server_address+assgWikiTitle+':'+candWikiName, headers={'Connection':'close'})
									if r_pmi.status_code == 200 and 'nan' not in r_pmi.text:
										ED_pmi = float(r_pmi.text)
										#if ED_wlm != 0.0 :
											#print 'cand %s assgEntity %s ED_wlm %0.8f ED_pmi %0.8f '%(candWikiName, getTitle(assgEntity), ED_wlm, ED_pmi)
								except ConnectionError as e: 
									print e
									EDServerInError = True											
								#print 'rhoIn = %0.6f. rhoOut = %0.6f. ED_wlm = %0.6f. ' %( rhoIn, rhoOut, ED_wlm)
								'''
								if rhoIn == -1.0:
									rhoIn = rel2(candInlinks, assgWikiTitle)
									if rhoOut == -1.0:
										rhoOut = relReverse2(candOutlinks, assgWikiTitle)
									LinkDb.WLM.insert({'E1':assgEntity, 'E2':candId, 'WLMin':rhoIn, 'WLMout':rhoOut}); #print 'relReverse stored for %s = %0.8f' %(str(Pb)+'#'+str(candId), WLM_out)
								else :
									if rhoOut == -1.0:
										rhoOut = relReverse2(candOutlinks, assgWikiTitle)
										LinkDb.WLM.insert({'E1':assgEntity, 'E2':candId, 'WLMin':rhoIn, 'WLMout':rhoOut}); #print 'rel stored for %s = %0.8f' %(str(assgEntity)+'#'+str(candId), rhoIn)
									else :
										reusedRelCount = reusedRelCount + 1
								'''
								rhoIn = rel2(candInlinks, assgWikiTitle)
								rhoOut = relReverse2(candOutlinks, assgWikiTitle)
								rhoDense = 0.0
								if assgEntity in assignmentPMIlists:
									if candWikiName in candidatePMIlists:
										#rhoDense = relDense2(candInlinks, candWikiName, getTitle(assgEntity))
										rhoDense = relDense3(candInlinks, candWikiName, assgWikiTitle, assgEntity); #print 'RhoDense3 %0.6f'%(rhoDense) #rhoDense = relDense3(candInlinks, candWikiName, assgWikiTitle, assgEntity);
								LinkDb.WLM.insert({'E1':assgEntity, 'E2':candId, 'WLMin':rhoIn, 'WLMout':rhoOut, 'WLMdense':rhoDense }); 
								#print 'rel stored for %s = %0.8f, %0.8f, %0.8f' %(str(assgEntity)+'#'+str(candId), rhoIn, rhoOut, rhoDense)
								
								rho = rho + rhoIn
								rho_out = rho_out + rhoOut
								rho_dense = rho_dense + rhoDense
								rho_ED_wlm = rho_ED_wlm + ED_wlm
								rho_ED_pmi = rho_ED_pmi + ED_pmi
							#print 'end', len(assignments)
							rho = 1.0 * rho / len(assignments) #Normalization
							rho_out = 1.0 * rho_out / len(assignments)
							rho_dense = 1.0 * rho_dense / len(assignments)
							rho_ED_wlm = 1.0 * rho_ED_wlm / len(assignments)
							rho_ED_pmi = 1.0 * rho_ED_pmi / len(assignments)
							#print 'rhoInlinks for candidte %s is %0.6f. rhoOutlinks is %0.6f. rho_dense is %0.6f. rho_ED_wlm is %0.6f. ReusedRelCount %d' %(candWikiName, rho, rho_out, rho_dense, rho_ED_wlm, reusedRelCount)
						else:
							print 'Found no assignments: candidate %s' %(candWikiName)
				else:
					print 'Inadequate inlinks or outlinks for candidate %s' %( candWikiName) #print 'Inadequate inlinks %d or outlinks %d for candidate %s' %(len(candInlinks), len(candOutlinks), candWikiName)

			FNUM = FNUM + 1 #F_6 WLM inlinks
			if (CollationMethod): # TAGME method
				f_list.append(str(FNUM)+':'+str(REL)) 
			else : # 2-step method
				f_list.append(str(FNUM)+':'+str(rho))

			FNUM = FNUM + 1 #F_7 WLM outlinks
			if (CollationMethod): # TAGME method
				f_list.append(str(FNUM)+':'+str(REL_out)) 
			else : # 2-step method
				f_list.append(str(FNUM)+':'+str(rho_out))
			#print 'time#4'; print datetime.datetime.now()

			FNUM = FNUM + 1 #F_8 ED by WLM 
			if (CollationMethod): # TAGME method
				f_list.append(str(FNUM)+':'+str(REL_ED_wlm)) 
			else : # 2-step method
				f_list.append(str(FNUM)+':'+str(rho_ED_wlm))
			#print 'time#4'; print datetime.datetime.now()

			FNUM = FNUM + 1 #F_9 ED by PMI 
			if (CollationMethod): # TAGME method
				f_list.append(str(FNUM)+':'+str(REL_ED_pmi)) 
			else : # 2-step method
				f_list.append(str(FNUM)+':'+str(rho_ED_pmi))

			FNUM = FNUM + 1 #F_10 WLM dense
			if (CollationMethod): # TAGME method
				f_list.append(str(FNUM)+':'+str(REL_dense)) 
			else : # 2-step method
				f_list.append(str(FNUM)+':'+str(rho_dense))				

			threadLock.acquire()
			Flist[candId] = f_list
			threadLock.release()
			#print ' cand : %d #f = %d, %s' %( candCount, FNUM, ','.join(f_list))
			#print('CAND Done in '+str(((datetime.datetime.now()-start).total_seconds())/60)+' minutes.')


def writeFeatures(datasetFlag, trainFlag, mentionCandList,Flist,pageCandPro,edQueuries):
	
	fOut = openOutputFile(datasetFlag, trainFlag);	
	for mention in mentionCandList:	
		for candId in mentionCandList[mention]:
			try:
			#if candId in Flist: #error case of no wikiId for candWikiPage
				f_list = Flist[candId]
				if len(f_list) != 13 : # 10 features +label + qId + candId
					print 'WRONG f_list :: mention %s cand %d' %(mention, candId)
					print f_list
				else : 
					if candId in pageCandPro:
						f_list.append('11:'+str(pageCandPro[candId])) #F_9
					else :
						f_list.append('11:0')
					f_list.append('12:'+str(1.0*1/len(mentionCandList[mention]))) #F_10 reciprocal of |candidates| for normalization
					#print "OUTPUT %s\n"%(','.join(f_list))
					fOut.write("%s\n"%(','.join(f_list)))
			except KeyError, e:
				print 'Raised a KeyError :: mention %s cand %d :: reason %s' %(mention, candId, str(e))

	fOut.close()
	print 'FILED : %s. Total mentions = %d, candidates = %d, time = %0.2f minutes'%(fOut.name, len(mentionCandList), len(Flist), ((datetime.now()-start_time).total_seconds())/60)

	#fQ = open(EDqueriesFile, 'a');	
	#for query in edQueuries:
	#fQ.write("%s\n"%(':::'.join(edQueuries)))
	#fQ.close()
	#print 'ED queries logged to  : %s. Added pairs = %d '%(fQ.name, len(edQueuries))		

def main():
	global datasetFlag, trainflag, restartFromQid, PMI_server_address
	
	if len(sys.argv) < 3:
		print("Missing arguments.\nFORMAT : python createTrainData.py <AIDA or TAC> <Train or Test> <0 or <start from Qid> <<PMI list server port>>");#<<TAC dataset split number>>");
		sys.exit();	
	#Input validation
	if sys.argv[1] == 'AIDA':
		datasetFlag = True
	elif sys.argv[1] == 'TAC':
		datasetFlag = False
	else :
		print("<TAC or AIDA> : Wrong argument.\nFORMAT : python createTrainData.py <AIDA or TAC> <Train or Test> <0 or Number> <<2338>>");
		sys.exit();

	if sys.argv[2] == 'Train':
		trainFlag = True
	elif sys.argv[2] == 'Test':
		trainFlag = False
	else :
		print("<Test or Train> : Wrong argument.\nFORMAT : python createTrainData.py <AIDA or TAC> <Train or Test> <0 or Number> ");
		sys.exit();

	if sys.argv[3] != '0':
		if (datasetFlag): #AIDA
			restartFromQid = int(sys.argv[3])
		else:
			restartFromQid = sys.argv[3]

	if len(sys.argv) == 5:
		PMI_server_address = 'http://10.16.32.104:'+sys.argv[4]+'/pmi/list?node='	

	loadVocab()

	if (datasetFlag): #AIDA
		if (trainFlag):
			sourceDir = AIDAsourceCandTrain 
		else:
			sourceDir = AIDAsourceCandTest 
		for fCan in listdir(sourceDir):
			if restartFromQid != 0:
				if int(fCan) != restartFromQid:
					continue
			with open(join(sourceDir,fCan), 'r') as fc:
				restartFromQid = 0
				print 'processing Query id %s'%fCan
				processFile(fc, datasetFlag, trainFlag)
			if EDServerInError or PMIServerInError:
				print "QUIT : EDServer or PMIlistServer is OOS: Processed till %s of %s" %(qId, fCan)
				break;

	else : #TAC
		if trainFlag:
			sourceFile = TACsourceCandTrain #sourceFile = TACsourceCandTrain+sys.argv[3]
		else:
			sourceFile = TACsourceCandTest #sourceFile = TACsourceCandTest+sys.argv[3]
		with open(sourceFile,'r') as fc:
			processFile(fc, datasetFlag, trainFlag)		
			print 'Finished processing %s'%(sourceFile)


if __name__ == "__main__":
	main();
	
#print(getTitle(20853148))
#print(getInlinks('Alpha_compositing'))#print(getPageId('Alpha_compositing'))
#print(getInlinks('Array_data_structure'))

'''
r = requests.get('http://10.16.34.115:1337/ED/hi:hello')
print (r)
print (r.text)
print (r.content)
'''

#getEntityPrior(321544) #print '%0.6f ' %lp('bangalore')
#text = 'I am in ; till _ only Bangalore Karnataka'
#print mentions(text)
#print getPageProbability('delhi')
'''
context = "2016 BRICS summit will be held in Banavali. Preperations for the summit is underway. The delegates will be housed in hotels in Cavelossium." # season training regimen for road-bike racers, has a storied Cyclocross, invented nearly 100 years ago in Europe as an off- tradition and a worldwide following. Internationally, professional competition is administered by the Union Cycliste Internationale, the Swiss organization that also oversees the from Maine to Colorado to California. USA Cycling's season-ending held each year nationwide, many organized into regional series, Cyclocross National Championships this year are in Kansas City, Kan., Dec. 13 to 16.  The course, set on the grounds of the National Sports Center athletics campus, twisted and turned on grass at first, with little orange flags and police tape strung to sequester the "
mention = 'Banavali' #'National Sports'
#contextSize = 10
context = conciseContext(mention, context, contextSize)
for m in mentions(context):
	if mention in m:
		print 'Found %s in %s' %(mention , m)

#candInlinks = getInlinks('Barack_Obama')
#WLM_in = rel2(candInlinks, getTitle(5422207))
print 'WLM_in %0.6f'%WLM_in
'''
'''
anchor1 = 'Barack_Obama'
anchor2 = 'Michelle_Obama'
loadVocab() 
#print getPMIentities(anchor2)
#relDense(anchor1, anchor2)
rel(anchor1, anchor2)
'''
'''
#rel calculation  for 965005#57654 
candWikiName = getTitle(965005)
candInlinks = getInlinks(candWikiName)
loadVocab()
rhoDense = relDense3(candInlinks, candWikiName, getTitle(57654), 57654)
print rhoDense
'''