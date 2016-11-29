#Find the popular and tail entities in wikipedia
#Read TAC and AIDA datasets. Find max, avg and min inlinked entities.
from os import listdir
from pymongo import MongoClient
from os.path import join
import csv
from datetime import datetime

#DATABASE SETUP
dbIP = '10.16.32.103' #db on momo 104' #db on dosa 
dbPort = 27017
client = MongoClient(dbIP, dbPort)
#client = MongoClient(); #db on localhost 
LinkDb = client.wikilinks

#Directory for entities and candidates source
#sourceCandDir ='/media/New Volume/Datasets/'#PPRforNED-master/AIDA_candidates' #lenovo laptop
sourceCandDir ='/scratch/home/priya/' #dosa
AIDAsourceCandTest =sourceCandDir+'AIDAforNED/PART_1001_1393'
AIDAsourceCandTrain =sourceCandDir+'AIDAforNED/PART_1_1000'
TACsourceCandTest =sourceCandDir+'TACforNED/Test.csv'
TACsourceCandTrain =sourceCandDir+'TACforNED/Train.csv'
inlinkMapFile = '/scratch/home/priya/inlinkMap.txt' #map of query entities to thier inlink counts

maxInl, minInl, entityCount, avgInl = 0,0,0,0.0 # max, avg and min inlinked entities.
pageInlinkMap = {}
start_time = datetime.now()
entInlinkMap = {}
Tail_Threshold = 1000
CountMap = {'tail':0, 'head':0, 'nil':0, 'no_entry':0}

def populateInlinkMap(fC):
	lineC = 0
	for line in fC:
		lineC = lineC + 1
		if str.startswith(line, "ENTITY"):
			if "url:http://en.wikipedia.org/wiki/" in line :
				entWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:].strip(); #print 'Entity wiki name : %s' %(entWikiName)
				if entWikiName not in pageInlinkMap :
					pageInlinkMap[entWikiName] = len(getInlinks(entWikiName))
		if lineC%1000 == 0:
			print 'Populated %d in inlinkMap' %(len(pageInlinkMap))

def getHeadTail(fC):
	tailCount, headCount = 0, 0
	global CountMap
	for line in fC:
		if str.startswith(line, "ENTITY"):
			if "url:http://en.wikipedia.org/wiki/" in line :
				entWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:].strip(); #print 'Entity wiki name : %s' %(entWikiName)
				if entWikiName in entInlinkMap :
					if int(entInlinkMap[entWikiName]) == 0:
						CountMap['nil'] += 1
					elif int(entInlinkMap[entWikiName]) > 0 and int(entInlinkMap[entWikiName]) <= Tail_Threshold :
						CountMap['tail'] += 1
					else:
						CountMap['head'] +=1
				else:
					CountMap['no_entry'] += 1	
	#print ' head %d, tail %d, NIL %d, no_InlinkEntry %d' %( CountMap['head'], CountMap['tail'], CountMap['nil'], CountMap['no_entry'])

def inlinkStatics():
	global maxInl, minInl, entityCount, avgInl
	for page in pageInlinkMap:
		inlinkCount = pageInlinkMap[page]
		if inlinkCount > maxInl :
			maxInl = inlinkCount
		elif inlinkCount < minInl :
			minInl = inlinkCount
		avgInl = avgInl + inlinkCount
		entityCount = entityCount + 1
		if entityCount%1000 == 0 :
			print 'maxIn %d, minIn %d, entityCount %d, avgIn %.4f' %(maxInl, minInl, entityCount, 1.0*avgInl/entityCount)

def getInlinks (pagename):
	inlinks = []
	linksReturned = LinkDb.inlinks.find({'page':pagename},{'link':1, '_id':0})
	for doc in linksReturned:
		inlinks.extend(doc['link'])
	return inlinks


def loadInlinkMap():
	fp = open (inlinkMapFile, 'r')
	for line in fp:
		#line.strip("{").strip("}")
		for row in line.split(", "):
			entity = row.split(':')[0].strip('"').strip("'").decode('utf-8')
			entInlinkMap[entity] = int(row.split(':')[1].strip().strip('"').strip("'"))
			#print(row.split(':')[0].strip('"').strip("'") ,row.split(':')[1].strip())
	print 'Inlink counts for %d' %(len(entInlinkMap))

#rewrites : call with caution
def writeInlinkMap():
	fo = open ('inlinkMap2.txt', 'w')
	fo.write(str(entInlinkMap))
	fo.close()	

def updateInlinkMap():
	for directory in [AIDAsourceCandTest,AIDAsourceCandTrain]: #AIDA	
		for fCan in listdir(directory):
			with open(join(directory,fCan), 'r') as fC:
				populateInlinkMap(fC)
	inlinkStatics()
	print 'AIDA entities = %d ' %(len(pageInlinkMap))
	print 'AIDA : maxIn %d, minIn %d, entityCount %d, avgIn %.4f' %(maxInl, minInl, entityCount, 1.0*avgInl/entityCount)
	'''
	for datafile in [TACsourceCandTest, TACsourceCandTrain]: #TAC
		with open(datafile,'r') as fc:
			populateInlinkMap(fc)
	inlinkStatics()
	print 'TAC entties = %d ' %(len(pageInlinkMap)) #print 'TAC queries = %d , missed context = %d' %(len(elTypes), missingContextCount)
	print 'AIDA and TAC : maxIn %d, minIn %d, entityCount %d, avgIn %.4f' %(maxInl, minInl, entityCount, 1.0*avgInl/entityCount)
	'''
	fo = open ('inlinkMap.txt', 'a')
	fo.write(str(pageInlinkMap))
	fo.close()

	with open('inlinkMap.csv', 'wb') as fw:
		w = csv.DictWriter(fw, pageInlinkMap.keys() )
		w.writeheader()
		w.writerow(pageInlinkMap)

	print 'This program ran in %0.2f minutes'%(((datetime.now()-start_time).total_seconds())/60)

#report number of tail and head entities in TAC and CoNLL
def tailHeadCounts():
	global CountMap
	for directory in [AIDAsourceCandTest,AIDAsourceCandTrain]: #AIDA	
		for fCan in listdir(directory):
			with open(join(directory,fCan), 'r') as fC:
				getHeadTail(fC)
		print 'AIDA : %s, head %d, tail %d, NIL %d, no_InlinkEntry %d' %(directory, CountMap['head'], CountMap['tail'], CountMap['nil'], CountMap['no_entry'])
		CountMap = {'tail':0, 'head':0, 'nil':0, 'no_entry':0}
	
	for datafile in [TACsourceCandTest, TACsourceCandTrain]: #TAC
		with open(datafile,'r') as fc:
			getHeadTail(fc)
		print 'TAC : %s, head %d, tail %d, NIL %d, no_InlinkEntry %d' %(datafile, CountMap['head'], CountMap['tail'], CountMap['nil'], CountMap['no_entry'])
		CountMap = {'tail':0, 'head':0, 'nil':0, 'no_entry':0}
	
#main

inlinkMapFile = '/scratch/home/priya/inlinkMap.txt'
entInlinkMap = {}
loadInlinkMap()
print ' No inlinks = %d ' %(len(entInlinkMap))
tailHeadCounts()

'''
#Update inlinkMap.txt with missing queries
loadInlinkMap()

missingEntities = ['Santos,_S\xc3\xa3o_Paulo', 'Man\xc3\xa1', 'Man\xc3\xa1', 'M\xc3\xa1laga', 'Cecilia_Malmstr\xc3\xb6m', 'Cecilia_Malmstr\xc3\xb6m', 'M\xc3\xa1laga', 'M\xc3\xa1laga', 'Santos,_S\xc3\xa3o_Paulo', 'Telef\xc3\xb3nica_Europe', 'Lothar_Matth\xc3\xa4us', "People's_Party_\xe2\x80\x93_Movement_for_a_Democratic_Slovakia", 'Aleksander_Kwa\xc5\x9bniewski', 'Clube_Atl\xc3\xa9tico_Mineiro', 'Bah\xc3\xada_Blanca', 'Ricardo_Pel\xc3\xa1ez', 'Jos\xc3\xa9_Luis_Caminero', 'Zoran_Savi\xc4\x87', 'Iv\xc3\xa1n_Zamorano', 'V\xc3\xa1clav_Havel', 'Bogot\xc3\xa1', 'D\xc3\xa9si_Bouterse', 'Alain_Jupp\xc3\xa9']
for mEntity in missingEntities:
	if "\\x" in r"%r" %mEntity:
		mEntity = r"%r" %mEntity
		mEntity = mEntity.strip('"').strip("'")
		print mEntity
		if mEntity in entInlinkMap:
			print 'Ha'

#print entInlinkMap.keys()
'''

'''
for mEntity in missingEntities:
	if mEntity not in entInlinkMap:
		#print mEntity
		entInlinkMap[mEntity] = len(getInlinks(mEntity))
print 'Now inlinks for %d ' %(len(entInlinkMap))
writeInlinkMap()

#Testing the updation
loadInlinkMap()
print '1. No inlinks = %d ' %(len(entInlinkMap))
inlinkMapFile = '/scratch/home/priya/inlinkMap2.txt'
entInlinkMap = {}
loadInlinkMap()
print '2. No inlinks = %d ' %(len(entInlinkMap))
'''

