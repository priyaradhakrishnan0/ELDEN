import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

import os, re
from os import listdir,path
from os.path import  join
import createTrainData
import random
import csv

#TACtestLinks = "/media/New Volume/Datasets/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.1/data/tac_2010_kbp_evaluation_entity_linking_query_types.tab" #lenovo laptop
#TACtrainLinks = "/media/New Volume/Datasets/TAC_2009_KBP_Gold_Standard_Entity_Linking_Entity_Type_List/data/Gold_Standard_Entity_Linking_List_with_Entity_Types.tab"
#TACtrainLinks = "/scratch/home/priya/TACforNED/tac_2009_Gold_Standard_Entity_Linking_List_with_Entity_Types.tab" #dosa
TACtestLinks = "/scratch/home/priya/TACforNED/tac_2010_kbp_evaluation_entity_linking_query_types.tab"
AIDAsourceCandTest ='/scratch/home/priya/AIDAforNED/PART_1001_1393'
TACsourceCandTest ='/scratch/home/priya/TACforNED/Test.csv'
#start EA
Tail_Threshold = 1200 #1000 #number of inlinks to an entity that decides it is popular or tail.
CommonRareSenseThreshold = 0.75
inlinkMapFile = '/scratch/home/priya/inlinkMap.txt' #map of query entities to thier inlink counts
elTypes = {}
elContexts = {}
elQueries = {}
elCandidate = {} #candId : candWikiName for AIDA
elMultiEntities = {} #qId : [enWikiName]
elEntities = {} #qId : enWikiName
elMentions = {} #qId :mention(s)
entInlinkMap = {}
querySenseMap = {} #map of qid to priProbmap
CommonSenseQuery = 0
CommonSenseCand = 0
CommonSenseCandMissed = 0
CommonRare = False #EA on common and rare senses turned ON(True) and OFF(False)
reportParameter = '' #Set this to TP to generate feature report on TP. Set to TN,FP and FN as needed.
#end EA

#Hyperparameters
NumFeatures = 13 #with ED_wlm , ED_pmi and WLMdense

noInlinkQueryCandPairs = [] #el entity pairs with one of the entities is no-inlink
def readNolinkPairs(filename):
	with open(filename, 'wb') as f:
		for line in f.readlines():
			noInlinkQueryCandPairs[line.split('###')[0]] = line.split('###')[1]
	print 'noInlinkQueryCandPairs loaded = %d' %(len(noInlinkQueryCandPairs))

def populateELmaps(datasetFlag):
	if datasetFlag : #AIDA	
		for fCan in listdir(AIDAsourceCandTest):
			with open(join(AIDAsourceCandTest,fCan), 'r') as fC:
				populateContests(fC, datasetFlag)
		print 'AIDA queries = %d ' %(len(elQueries))
	else : #TAC
		with open(TACsourceCandTest,'r') as fc:
			populateContests(fc, datasetFlag)

		missingContextCount = 0
		with open (TACtestLinks, 'r') as fT:
			for line in fT:
				elTypes[line.split()[0].strip()] = line.split()[2].strip()
				#if line.split()[0].strip() not in elContexts: #empirically found missingContextCount = 1230 which corresponds to 1230 NIL queries in 2010 TAC query set
				#	print 'missing context for %s' %(line.split()[0].strip())
				#	missingContextCount = missingContextCount + 1
		'''
		print 'TAC queries = %d ' %(len(elQueries)) #print 'TAC queries = %d , missed context = %d' %(len(elTypes), missingContextCount)
		print 'TAC multi mention queries = %d ' %(len(elMultiEntities))
	print 'Test entitries = %d '%(len(elEntities))
	print 'Test contexts = %d' %(len(elContexts))
	print 'Test types = %d ' %(len(elTypes))
	print 'Inlink counts for %d' %(len(entInlinkMap))
	print 'Test Mentions = %d' %(len(elMentions))
		'''
	#for o in elMentions: #for o in random.sample(elMentions, 3) :
	#	print o, elMentions[o]

#slow. Involves multiple db queries
def populateSenseMaps(fC, datasetFlag):
	global CommonSenseQuery
	for line in fC:
		if str.startswith(line, "ENTITY"):
			if datasetFlag : #AIDA
				entityMention = line[12:line.find('normalName:')].strip()
				qId = 'Q'+os.path.basename(fC.name)				
			else : #TAC
				entityMention = line[12:line.find('qid:')].strip()
				qId = line[line.find('qid:')+4:line.find('docId:')].strip() 
			querySenseMap[qId] = createTrainData.getPageProbability(entityMention) 

			if "url:http://en.wikipedia.org/wiki/" in line :
				entWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:].strip(); #print 'Entity wiki name : %s' %(entWikiName)
				entWikiId = createTrainData.getPageId(entWikiName)
				if entWikiId in querySenseMap[qId]:
					if querySenseMap[qId][entWikiId] > CommonRareSenseThreshold:
						CommonSenseQuery = CommonSenseQuery + 1 # Query links to its common sense
						#print ' Query %s Mention %s links to Common sense %s' %(qId, entityMention, entWikiName)

	#Common and Rare Sense queries
	print ' Query division by Common and Rare senses : Common sense = %d, Rare sense = %d' %(CommonSenseQuery, len(querySenseMap) - CommonSenseQuery)

def populateContests(fC, datasetFlag):
	#candList = []#used only for TAC
	for line in fC:
		if str.startswith(line, "ENTITY"):
			#if len(candList) > 0: #if qId is not None and len(candList) > 0:#qId from last entity
			#	elCandidate[qId] = candList
			#	candList = []#qId = None

			contextRead = True
			if datasetFlag : #AIDA
				entityMention = line[12:line.find('normalName:')].strip()
				qId = 'Q'+os.path.basename(fC.name)
			else : #TAC
				entityMention = line[12:line.find('qid:')].strip()
				qId = line[line.find('qid:')+4:line.find('docId:')].strip() 
				docId = line[line.find('docId:')+6:line.find('url:')].strip() 
			elQueries[qId]=line
			if qId in elMentions:
				elMentions[qId].append(entityMention)				
			else:
				elMentions[qId] = [entityMention] 

			if "url:http://en.wikipedia.org/wiki/" in line :
				entWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:].strip(); #print 'Entity wiki name : %s' %(entWikiName)
				if qId in elMultiEntities:
					elMultiEntities[qId].append(entWikiName)
				else:
					elMultiEntities[qId] = [entWikiName]
				elEntities[qId] = entWikiName

		elif str.startswith(line, "CANDIDATE"):
			candId = line[16:line.find('inCount:')].strip()
			if datasetFlag : #AIDA
				candWikiName = line[line.find('url:http://en.wikipedia.org/wiki/')+33:line.find('name:')].strip()
				elCandidate[candId] = candWikiName
			else : #TAC
				candWikiNameMatch = line.rfind('/');#candWikiNameMatch = re.match('(.*)url:http://en.wikipedia.org/wiki/([^\s]+)', line)
				candWikiName = line[candWikiNameMatch+1:].strip()
				#candList.append(candWikiName)
			
			typeInfo = line[line.find('predictedType:')+14:].strip()
			if typeInfo == 'GPE' or typeInfo == 'PER' or typeInfo == 'ORG':
				elTypes[candId] = typeInfo
		elif str.startswith(line, "CONTEXT"):
			if contextRead :
				context = line.strip().replace("CONTEXT\t", "")
				#contextSize = 15
				#context = createTrainData.conciseContext(entityMention, context, contextSize ) #
				elContexts[qId] = context
				contextRead = False
	#Tail and popular entities
	fp = open (inlinkMapFile, 'r')
	for line in fp:
		#line.strip("{").strip("}")
		for row in line.split(", "):
			entity = row.split(':')[0].strip('"').strip("'").decode('utf-8')
			entInlinkMap[entity] = int(row.split(':')[1].strip().strip('"').strip("'"))

def numpyCreate(dataset):
	#create numpy arrays for scikit learn.
	feat=[];
	target=[];
	for sample in dataset:
		content=sample.strip().split(',');
		f=[];
		for i in range(len(content)-1):
			#if (i== 1 or i==5) :
			#if not (i== 3 or i==4) : #skip string similarity features .
				#f.append(float(content[i+1].split(':')[1]));
			f.append(float(content[i+1]));
			#f.append(float(content[i]));
		feat.append(np.array(f));
		target.append(content[0])
		#target.append(content[len(content)-1])

	print 'feat samples = %d, label samples = %d, #features = %d '%(len(feat), len(target), len(feat[0]))
	'''
	Feature normalization
	'''
	#Method 1
	feat=preprocessing.scale(feat);

	#Method 2
	#feat=preprocessing.normalize(feat,norm='l2');

	return (feat,target);

def numpyCreate2(dataset, featureSize):
	#create numpy arrays for scikit learn.
	feat=[];
	target=[];
	for sample in dataset:
		content=sample.strip().split(',');
		f=[];
		for i in range(len(content)-1):
			if (i<featureSize) :
			#if not (i== 3 or i==4) : #skip string similarity features .
				#f.append(float(content[i+1].split(':')[1]));
				f.append(float(content[i+1]));
			#f.append(float(content[i]));
		feat.append(np.array(f));
		target.append(content[0])
		#target.append(content[len(content)-1])

	print 'feat samples = %d, label samples = %d, #features = %d'%(len(feat), len(target), len(feat[0]))
	'''
	Feature normalization
	'''
	#Method 1
	feat=preprocessing.scale(feat);

	#Method 2
	#feat=preprocessing.normalize(feat,norm='l2');

	return (feat,target);

def numpyCreate_EA(dataset): #, featureSize):
	#create numpy arrays for scikit learn.
	feat=[];
	target=[];
	track_train = []
	coherence_features = [6,7]
	ED_wlm = [8]
	ED_pmi = [9]
	WLMdense = [10]
	string_sim_features = [4,5]
	base_features = [1,2,3,11,12] 
	for sample in dataset:
		content=sample.strip().split(','); #print 'Contents = %d' %len(content)
		if len(content) >= NumFeatures: #without ED =10:#== featureSize : #12 for AIDA, 10 for TAC_current
			qId = content[0].split(':')[1]
			candId = content[1].split(':')[1]
			features = [content[i].split(':')[1] for i in range(2,len(content))]
			#print '#features = %d' %len(features)
			f=[];
			for i in range(1,len(features)): #-1):#correction for normalization of F9 #revert#
				if i in string_sim_features or i == 10: #if i in base_features or i in string_sim_features or i in coherence_features : #if not(i==10 or i==9 or i==8 or i in coherence_features): #if not (i in base_features or i in coherence_features or i==9 ): #if not (i in coherence_features or i in string_sim_features): #if not (i in coherence_features) #if not(i==7 or i==6 ):#F6=rho_in, F7=rho_out:
					f.append(float(features[i]));
			#f.append(1.0*1/float(features[len(features)-1]))#normalised F_9
			#print f
			feat.append(np.array(f));
			target.append(features[0])
			track_train.append(qId+':'+candId)

	#print 'feat samples = %d, label samples = %d, trackers = %d'%(len(feat), len(target), len(track_train))
	print 'feat samples = %d, label samples = %d, #features = %d, trackers = %d'%(len(feat), len(target), len(feat[0]), len(track_train))
	'''
	Feature normalization
	'''
	#Method 1
	feat=preprocessing.scale(feat);

	#Method 2
	#feat=preprocessing.normalize(feat,norm='l2');

	return (feat,target, track_train);

#create a list feature arrays from the dataset
def getFeatures(dataset): #, featureSize):
	F=[];
	for sample in dataset:
		content=sample.strip().split(','); # print 'Contents = %d' %len(content); print content
		if len(content) >= NumFeatures: #10 without ED #== featureSize : #12 for AIDA, 10 for TAC_current
			#qId = content[0].split(':')[1]
			#candId = content[1].split(':')[1]
			#label = content[2].split(':')[1]
			features = [float(content[i].split(':')[1]) for i in range(3,len(content))] 
			#features = [float(content[i].split(':')[1]) for i in range(3,len(content)-1)] #correction for normalization of F9 
			#features.append(1.0*1/float(content[len(content)-1].split(':')[1]))
		F.append(features)
	return F

def analyzeFeatures():
	if len(sys.argv)!=2:
		print("Missing arguments.\nFORMAT : python classify.py <Train_file/Test_file>");
		sys.exit();
		#Read the train file.
	train_file=sys.argv[1];
	train_dataset=[];
	with open(train_file) as f:
		train_dataset=f.readlines();

	feature_set = getFeatures(train_dataset)
	avj = [0.0] * NumFeatures
	wlm_count, ED_wlm_count, ED_pmi_count, WLMdense_count = 0, 0, 0, 0
	for i in range(len(feature_set)):
		#print '#f = %d'%(len(feature_set[i]))
		if feature_set[i][5] != 0 or feature_set[i][6] != 0:			#print feature_set[i]
			wlm_count = wlm_count + 1
		if feature_set[i][7] != 0 :			#print feature_set[i]
			ED_wlm_count = ED_wlm_count + 1
		if feature_set[i][8] != 0 :
			ED_pmi_count = ED_pmi_count + 1
		if feature_set[i][9] != 0 :
			WLMdense_count = WLMdense_count + 1			
		for j in range(NumFeatures-1):
			avj[j] = avj[j] + feature_set[i][j]

	AVG = [1.0*k/len(feature_set) for k in avj ]
	avgFeatStr = ''
	print 'Average Feature Values '
	for j in range(NumFeatures-1):
		avgFeatStr += str(j+1)+ ':'+ str(AVG[j]) +', ' #print '%d %0.4f' %(j+1, AVG[j])
	#print AVG
	print 'Total %d samples' %( len(feature_set))
	print 'Average feature values '+avgFeatStr
	print 'wlm was positive for %d samples, ED_wlm for %d, ED_pmi for %d and WLMdense for %d samples' %(wlm_count, ED_wlm_count, ED_pmi_count, WLMdense_count)


def classify_EA(train_set,test_set,datasetFlag):
	global entInlinkMap, CommonSenseCand, CommonSenseCandMissed
	#Classification algorithm.
	print 'Train set :'
	train_feat,train_target,track_train=numpyCreate_EA(train_set);#,12);#train_feat,train_target=numpyCreate(train_set);
	print 'Test set :'
	#test_feat,test_target=numpyCreate(test_set);
	test_feat,test_target,track_test=numpyCreate_EA(test_set);#,12);#test_feat,test_target=numpyCreate2(test_set,5); #TAC

	#Build the pipeline
	clf=Pipeline([('clf', RandomForestClassifier(max_features=2,n_estimators=100))]);#, verbose=3))]);
	#clf=Pipeline([('clf', ExtraTreesClassifier(max_features=3,n_estimators=100))]);
	#clf=Pipeline([('clf', AdaBoostClassifier(n_estimators=100))]);0
	#clf=Pipeline([('clf', GradientBoostingClassifier(n_estimators=10000, learning_rate=0.02, max_depth=4, random_state=0, verbose=3))]);

	#Training
	clf.fit(train_feat,train_target);

	#Testing
	predicted=clf.predict(test_feat);

	#print np.mean(predicted==test_target);	

	target_names=['0','1'];
	#print(metrics.confusion_matrix(test_target, predicted));
	print(metrics.classification_report(test_target, predicted,target_names=target_names));
	#print (metrics.precision_score(test_target, predicted, labels=None, pos_label='1', average='micro'))
	#print (metrics.precision_score(test_target, predicted, labels=None, pos_label='1', average='binary'))#macro'))
	#print(metrics.precision_recall_fscore_support(test_target, predicted,labels=None, pos_label='1', average=None));

	#Feature weights.
	print(clf.named_steps['clf'].feature_importances_)

	#EA start
	#datasetFlag = False #AIDA - True, TAC - False
	populateELmaps(datasetFlag)
	test_feature_set = getFeatures(test_set)
	TP, TN, FP, FN = 0,0,0,0
	TP_list, TN_list, FP_list, FN_list, TP_track, TN_track, FP_track, FN_track = {}, {}, {}, {}, {}, {}, {}, {}
	#F1, F2, F3, F4, F5, F6, F7, F8, F9 = 0,0,0,0,0,0,0,0,0
	avj_tp = [0.0] * NumFeatures #instantiate average list with 0.0
	avj_tn = [0.0] * NumFeatures
	avj_fp = [0.0] * NumFeatures
	avj_fn = [0.0] * NumFeatures
	Pmacro = {} # query to TP,TN,FP,FN counts map #AIDA
	PosNegCounts = {'TP':0,'TN':0,'FP':0,'FN':0} 
	TP, FP, TN, FN = 0,0,0,0
	TP_gpe, TP_per, TP_org, TN_gpe, TN_per, TN_org, FP_gpe, FP_per, FP_org, FN_gpe, FN_org, FN_per = 0,0,0,0, 0,0,0,0, 0,0,0,0
	TP_web, TP_news, TN_web, TN_news, FP_web, FP_news, FN_web, FN_news = 0,0,0,0, 0,0,0,0
	missingTypeCount = 0
	queryResultsMap = {}
	torsoDivisions = [0, 200, 400, 600, 800, 1000, 1200] #[0, 250, 500, 750, 1000] #torsoDivisions = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	torsoFeatures= dict.fromkeys(torsoDivisions)
	torsoPRF = dict.fromkeys(torsoDivisions)
	for i in range(0, len(torsoDivisions)):
		torsoFeatures[torsoDivisions[i]] = [0.0 for j in range(NumFeatures)]
		torsoPRF[torsoDivisions[i]] = {'TP':0,'TN':0,'FP':0,'FN':0}

	qId_stored = ''
	candRank = 0
	candLinkingMap = {} #map of candRank to linking status
	print 'test_track = %d, test_feature_set = %d' %(len(track_test), len(test_feature_set))

	if reportParameter == 'TP' or reportParameter == 'FP' or reportParameter == 'TN' or reportParameter == 'FN':
		csvfile = open('Report.csv','wb')
		fwriter = csv.writer(csvfile, delimiter=';')

	for i in range(len(track_test)):
		#doldTypeKey is candId
		if datasetFlag : #AIDA
			goldTypeKey = track_test[i].split(':')[1]#AIDA
			qId = 'Q'+track_test[i].split(':')[0]
		else :
			goldTypeKey = track_test[i].split(':')[0]#TAC
			qId = track_test[i].split(':')[0]

		if qId_stored != qId : #next query. Print out last qId
			queryResultsMap[qId_stored] = candLinkingMap
			qId_stored = qId
			candRank = 0
			candLinkingMap = {}

		if CommonRare:
			candId = track_test[i].split(':')[1]
			if candId in querySenseMap[qId]:
				if querySenseMap[qId][candId] > CommonRareSenseThreshold:
					CommonSenseCand = CommonSenseCand + 1 # Cand links to its common sense
					#print ' Cand %s links to Common sense' %(candId)
			else:
				CommonSenseCandMissed = CommonSenseCandMissed + 1


		candRank = 	candRank + 1
		if predicted[i]==test_target[i]:
			if predicted[i] == '1':
				candLinkingMap[candRank] = 1
			else:
				candLinkingMap[candRank] = 0				
		else:
			candLinkingMap[candRank] = 0

		qString = elQueries[qId] 
		if datasetFlag : #AIDA. Pmacro
			if qId in Pmacro:
				PosNegCounts = Pmacro[qId]
			else:
				PosNegCounts = {'TP':0,'TN':0,'FP':0,'FN':0} 

			if predicted[i]==test_target[i]:
				if predicted[i] == '1':			#TP = TP + 1
					PosNegCounts['TP'] = PosNegCounts['TP'] + 1
				else: #TN = TN + 1
					PosNegCounts['TN'] = PosNegCounts['TN'] + 1
			else:
				if predicted[i] == '1': #FP = FP + 1
					PosNegCounts['FP'] = PosNegCounts['FP'] + 1
				else: #FN = FN + 1 
					PosNegCounts['FN'] = PosNegCounts['FN'] + 1
			Pmacro[qId] = PosNegCounts

		else: #TAC Genre of query text
			if "docId:eng-" in qString: #web text
				QueryTextGenre = 'WEB'
			elif "_ENG_" in qString : # English News Text. Could be New York Times, Associated Press, Xinhua News 
				QueryTextGenre = 'NEWS'


		if goldTypeKey in elTypes: #if type info exists for this 
			goldType = elTypes[goldTypeKey]
			if predicted[i]==test_target[i]:
				if predicted[i] == '1':	
					#TP = TP + 1
					if goldType == 'ORG':
						TP_org = TP_org + 1
					elif goldType == 'PER':
						TP_per = TP_per + 1
					elif goldType == 'GPE':
						TP_gpe = TP_gpe + 1
		
				else: 
					#TN = TN + 1					
					if goldType == 'ORG':
						TN_org = TN_org + 1
					elif goldType == 'PER':
						TN_per = TN_per + 1
					elif goldType == 'GPE':
						TN_gpe = TN_gpe + 1
		
			else:
				if predicted[i] == '1': #Negative predicted as positive 
					if goldType == 'ORG':
						FP_org = FP_org + 1
					elif goldType == 'PER':
						FP_per = FP_per + 1
					elif goldType == 'GPE':
						FP_gpe = FP_gpe + 1
			
				else: 
			
					if goldType == 'ORG':
						FN_org = FN_org + 1
					elif goldType == 'PER':
						FN_per = FN_per + 1
					elif goldType == 'GPE':
						FN_gpe = FN_gpe + 1			
		else :
			missingTypeCount = missingTypeCount + 1
		
		#Generate report for TP/TN/FP/FN
		if predicted[i]==test_target[i]:
			if predicted[i] == '1':	
				if reportParameter == 'TP':
					writeRow = []
					writeRow.append(test_feature_set[i]) # TP_list[TP] = test_feature_set[i]					
					writeRow.append(elMentions[qId]) #TP_track[TP] = elMentions[qId]
					writeRow.append(qId) #TP_track[TP].append(qId)
					writeRow.append(elMultiEntities[qId])#TP_track[TP].append(elMultiEntities[qId])
					writeRow.append(len(elMultiEntities[qId]))#TP_track[TP].append('::')
					if datasetFlag :
						writeRow.append(elCandidate[goldTypeKey]) #TP_track[TP].append(elCandidate[goldTypeKey])
					else :
						writeRow.append(createTrainData.getTitle(int(track_test[i].split(':')[1])) ) #TP_track[TP].append(createTrainData.getTitle(int(track_test[i].split(':')[1])) )
					fwriter.writerow(writeRow)
					#csvfile.flush()
					#import pdb; pdb.set_trace()
					#Average feature values
					for j in range(len(test_feature_set[i])-1):
						avj_tp[j] = avj_tp[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							TP_web = TP_web + 1
						elif QueryTextGenre == 'NEWS':
							TP_news = TP_news + 1

				#Average feature value of torso queries
				mEntity = elEntities[qId] #To take care of raw dumping (not utf-8 encoded)
				if "\\x" in r"%r" %mEntity:
					mEntity = r"%r" %mEntity
					mEntity = mEntity.strip('"').strip("'"); #print mEntity				

				if mEntity in entInlinkMap:
					if int(entInlinkMap[mEntity]) >= 0 and int(entInlinkMap[mEntity]) <= Tail_Threshold :
						for l in range(0,len(torsoDivisions)-1):
							if int(entInlinkMap[mEntity]) in range(torsoDivisions[l], torsoDivisions[l+1]) :
								#print torsoDivisions[l], torsoDivisions[l+1], entInlinkMap[mEntity], linkedCount
								for j in range(len(test_feature_set[i])-1):
									torsoFeatures[torsoDivisions[l]][j] = torsoFeatures[torsoDivisions[l]][j] + test_feature_set[i][j]
								torsoPRF[torsoDivisions[l]]['TP'] += 1

				TP = TP + 1							
			
			else: 
				if reportParameter == 'TN':
					writeRow = []
					writeRow.append(test_feature_set[i]) # TN_list[TN] = test_feature_set[i] 
					writeRow.append(elMentions[qId]) #TN_track[TN] = elMentions[qId]
					writeRow.append(qId) #TN_track[TN].append(qId)
					writeRow.append(elMultiEntities[qId]) #TN_track[TN].append(elMultiEntities[qId])
					writeRow.append(len(elMultiEntities[qId]))#TN_track[TN].append('::')
					#if datasetFlag : #>>>DEBUG HERE>>
					#	TN_track[TN].append(elCandidate[goldTypeKey])
					#else :
					#	TN_track[TN].append( createTrainData.getTitle(int(track_test[i].split(':')[1])) )
					fwriter.writerow(writeRow)
					#Average feature values
					for j in range(len(test_feature_set[i])-1):
						avj_tn[j] = avj_tn[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							TN_web = TN_web + 1
						elif QueryTextGenre == 'NEWS':
							TN_news = TN_news + 1

				#P R F value of torso queries
				mEntity = elEntities[qId] #To take care of raw dumping (not utf-8 encoded)
				if "\\x" in r"%r" %mEntity:
					mEntity = r"%r" %mEntity
					mEntity = mEntity.strip('"').strip("'"); #print mEntity	
				if mEntity in entInlinkMap:
					if int(entInlinkMap[mEntity]) >= 0 and int(entInlinkMap[mEntity]) <= Tail_Threshold :
						for l in range(0,len(torsoDivisions)-1):
							if int(entInlinkMap[mEntity]) in range(torsoDivisions[l], torsoDivisions[l+1]) :
								torsoPRF[torsoDivisions[l]]['TN'] += 1

				TN = TN + 1							
			
		else:
			if predicted[i] == '1': #Negative predicted as positive
				if reportParameter == 'FP':
					writeRow = []
					writeRow.append(test_feature_set[i]) #FP_list[FP] = test_feature_set[i] 
					writeRow.append(elMentions[qId]) #FP_track[FP] = elMentions[qId]
					writeRow.append(qId) #FP_track[FP].append(qId)
					writeRow.append(elMultiEntities[qId])#FP_track[FP].append(elMultiEntities[qId])
					writeRow.append(len(elMultiEntities[qId])) #FP_track[FP].append('::')
					if datasetFlag :
						writeRow.append(elCandidate[goldTypeKey]) #FP_track[FP].append( elCandidate[goldTypeKey] )
					else :
						writeRow.append(createTrainData.getTitle(int(track_test[i].split(':')[1])) ) #FP_track[FP].append( createTrainData.getTitle(int(track_test[i].split(':')[1])) )
					fwriter.writerow(writeRow)

					for j in range(len(test_feature_set[i])-1):
						avj_fp[j] = avj_fp[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							FP_web = FP_web + 1
						elif QueryTextGenre == 'NEWS':
							FP_news = FP_news + 1

				#P R F value of torso queries
				mEntity = elEntities[qId] #To take care of raw dumping (not utf-8 encoded)
				if "\\x" in r"%r" %mEntity:
					mEntity = r"%r" %mEntity
					mEntity = mEntity.strip('"').strip("'"); #print mEntity	
				if mEntity in entInlinkMap:
					if int(entInlinkMap[mEntity]) >= 0 and int(entInlinkMap[mEntity]) <= Tail_Threshold :
						for l in range(0,len(torsoDivisions)-1):
							if int(entInlinkMap[mEntity]) in range(torsoDivisions[l], torsoDivisions[l+1]) :
								torsoPRF[torsoDivisions[l]]['FP'] += 1
				FP = FP + 1
		
			else: 
				#positive predicted as negative
				if reportParameter == 'FN':
					writeRow = []
					writeRow.append(test_feature_set[i]) #FN_list[FN] = test_feature_set[i] 
					writeRow.append(elMentions[qId]) #FN_track[FN] = elMentions[qId]
					writeRow.append(qId) #FN_track[FN].append(qId)
					writeRow.append(elMultiEntities[qId])#FN_track[FN].append(elMultiEntities[qId])
					writeRow.append(len(elMultiEntities[qId])) #FN_track[FN].append('::')
					if datasetFlag :
						writeRow.append(elCandidate[goldTypeKey]) #FN_track[FN].append( elCandidate[goldTypeKey] )
					else :
						writeRow.append(createTrainData.getTitle(int(track_test[i].split(':')[1])) ) #FN_track[FN].append( createTrainData.getTitle(int(track_test[i].split(':')[1])) )
					fwriter.writerow(writeRow)
						
					for j in range(len(test_feature_set[i])-1):
						avj_fn[j] = avj_fn[j] + test_feature_set[i][j]						
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							FN_web = FN_web + 1
						elif QueryTextGenre == 'NEWS':
							FN_news = FN_news + 1

				#P R F value of torso queries
				mEntity = elEntities[qId] #To take care of raw dumping (not utf-8 encoded)
				if "\\x" in r"%r" %mEntity:
					mEntity = r"%r" %mEntity
					mEntity = mEntity.strip('"').strip("'"); #print mEntity	
				if mEntity in entInlinkMap:
					if int(entInlinkMap[mEntity]) >= 0 and int(entInlinkMap[mEntity]) <= Tail_Threshold :
						for l in range(0,len(torsoDivisions)-1):
							if int(entInlinkMap[mEntity]) in range(torsoDivisions[l], torsoDivisions[l+1]) :
								torsoPRF[torsoDivisions[l]]['FN'] += 1
				FN = FN + 1			
			

	#print 'Missing cand type for %d' %(missingTypeCount)
	#TP = TP_per + TP_gpe + TP_org
	#TN = TN_per + TN_gpe + TN_org
	#FP = FP_per + FP_gpe + FP_org
	#FN = FN_per + FN_gpe + FN_org
	if reportParameter == 'TP' or reportParameter == 'FP' or reportParameter == 'TN' or reportParameter == 'FN':
		csvfile.close()

	#per query analysis
	rank = 1
	Result = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0}
	multiLinkedQuery, torsoQuery_linked, torsoQuery_unlinked, tailQuery_linked, tailQuery_unlinked, PopularQuery_linked, PopularQuery_unlinked , missingQueryWikiName = 0,0,0,0,0,0,0,0
	torsoQuery= dict.fromkeys(torsoDivisions)
	for i in range(0, len(torsoDivisions)):
		torsoQuery[torsoDivisions[i]] = {'linked':0, 'unlinked':0}
	missingEntInlink = []
	#print '%d query results populated' %(len(queryResultsMap))

	#for l in range(0,len(torsoDivisions)-1):
	#	print l, torsoDivisions[l], torsoDivisions[l+1], torsoQuery[torsoDivisions[l]]['linked'], torsoQuery[torsoDivisions[l]]['unlinked']

	for query in queryResultsMap:
		#print queryResultsMap[query]
		for rank in range(1,21):
			if len(queryResultsMap[query]) >= rank:
				if queryResultsMap[query][rank] == 1 :
					Result[rank] = Result[rank] + 1

		#sanity check- Does a query have multiple links
		#if len([k for k in range(1,len(queryResultsMap[query])) if queryResultsMap[query][k]==1]) > 1 :
		#	multiLinkedQuery = multiLinkedQuery  + 1 #empirically found out as 0 in TAC and positive in AIDA
			#Popular versus tail queries
		
		if query in elEntities:
			#print 'query = %s, enWikiname = %s' %(query, elEntities[query])
			#print "===================================="
			#print entInlinkMap
			#print "===================================="
			mEntity = elEntities[query] #To take care of raw dumping (not utf-8 encoded)
			if "\\x" in r"%r" %mEntity:
				mEntity = r"%r" %mEntity
				mEntity = mEntity.strip('"').strip("'"); #print mEntity				

			if mEntity in entInlinkMap:
				#print 'query = %s, inlinks = %s' %(query, entInlinkMap[elEntities[query]])
				linkedCount = len([k for k in range(1,len(queryResultsMap[query])) if queryResultsMap[query][k]==1])
				'''
				if int(entInlinkMap[mEntity]) == 0 : 
					if linkedCount >= 1 :
						tailQuery_linked = tailQuery_linked + 1
					else:
						tailQuery_unlinked = tailQuery_unlinked + 1
				'''
				if int(entInlinkMap[mEntity]) > Tail_Threshold : 
					if linkedCount  >= 1 :
						PopularQuery_linked = PopularQuery_linked + 1
					else:
						PopularQuery_unlinked = PopularQuery_unlinked + 1
				elif int(entInlinkMap[mEntity]) >= 0 and int(entInlinkMap[mEntity]) <= Tail_Threshold :
					#torso queries
					'''
					print 'TORSO'
					if linkedCount  >= 1 :
						torsoQuery_linked +=  1
						if int(entInlinkMap[mEntity]) in range( torsoDivisions[0], torsoDivisions[1]):
							torsoQuery[torsoDivisions[0]]['linked'] += 1
							print 'TORSO 1'
					else:
						torsoQuery_unlinked += 1
					'''

					#l=0
					#print torsoDivisions[l], torsoDivisions[l+1], entInlinkMap[mEntity], linkedCount
					for l in range(0,len(torsoDivisions)-1):
						if int(entInlinkMap[mEntity]) in range(torsoDivisions[l], torsoDivisions[l+1]) :
							#print torsoDivisions[l], torsoDivisions[l+1], entInlinkMap[mEntity], linkedCount
							if linkedCount >= 1:
								torsoQuery[torsoDivisions[l]]['linked'] = torsoQuery[torsoDivisions[l]]['linked'] +  1
								#print torsoDivisions[l], torsoDivisions[l+1], entInlinkMap[mEntity]
							else:
								torsoQuery[torsoDivisions[l]]['unlinked'] = torsoQuery[torsoDivisions[l]]['linked'] +  1
							#break;
					

			else:
				missingEntInlink.append(elEntities[query])
		else:
			missingQueryWikiName = missingQueryWikiName + 1

	#for l in range(0,len(torsoDivisions)-1):
	#	print l, torsoDivisions[l], torsoDivisions[l+1], torsoQuery[torsoDivisions[l]]['linked'], torsoQuery[torsoDivisions[l]]['unlinked']

	#Common and Rare Senses
	if CommonRare : #Common and Rare Sense candidates
		print ' Query division by Common and Rare senses : Common sense = %d, Rare sense = %d, CommonRareSenseThreshold = %0.4f' %(CommonSenseQuery, len(querySenseMap) - CommonSenseQuery, CommonRareSenseThreshold)
		print ' Candidate division by Common and Rare senses : Common sense = %d, Rare sense = %d, CommonSenseCandMissed = %d' %(CommonSenseCand, len(track_test) - CommonSenseCand, CommonSenseCandMissed)

	print 'Popularity in linking:'
	if PopularQuery_linked+PopularQuery_unlinked > 0:
		print 'Head queries - linked = %0.4f%%(%d) and unlinked = %0.4f%%(%d)' %(100.0*PopularQuery_linked/(PopularQuery_linked+PopularQuery_unlinked), PopularQuery_linked, 100.0*PopularQuery_unlinked/(PopularQuery_linked+PopularQuery_unlinked), PopularQuery_unlinked)
	
	print 'Torso queries'# : %%linked ; linked ; %%unlinked ; unlinked'
	for l in range(len(torsoDivisions)-1, -1, -1):
		if torsoQuery[torsoDivisions[l]]['linked']+torsoQuery[torsoDivisions[l]]['unlinked'] > 0 :
			
			#print '%d - %d ' %(torsoDivisions[l], torsoDivisions[l+1])
			#print '%0.4f' %(100.0 * torsoQuery[torsoDivisions[l]]['linked']/(torsoQuery[torsoDivisions[l]]['linked']+torsoQuery[torsoDivisions[l]]['unlinked']))
			torsoQuery_linked += torsoQuery[torsoDivisions[l]]['linked']
			torsoQuery_unlinked += torsoQuery[torsoDivisions[l]]['unlinked']
			#print torsoFeatures[torsoDivisions[l]]
			#if torsoQuery[torsoDivisions[l]]['linked'] > 0 :
			#	print [1.0*f/torsoQuery[torsoDivisions[l]]['linked'] for f in torsoFeatures[torsoDivisions[l]] ]

	
	torsoOutputString1, torsoOutputString2, torsoOutputString3 = '','',''
	T_sum = [0.0 for j in range(NumFeatures)]
	cumulatedTP, cumulatedFP, cumulatedFN = 0, 0, 0 #to report cumulative P,R and F in torso entities rather than absolute P,R and F when inlinks in 0-200 ...
	for l in range(0,len(torsoDivisions)): #for l in range(len(torsoDivisions)-1, -1, -1):
		torsoOutputString1 = torsoOutputString1 + str( 100.0 * torsoQuery[torsoDivisions[l]]['linked']/(torsoQuery_linked + torsoQuery_unlinked)) + ':' + str(torsoQuery[torsoDivisions[l]]['linked']) + ';'
		
		if torsoPRF[torsoDivisions[l]]['TP'] > 0 :
			cumulatedTP += torsoPRF[torsoDivisions[l]]['TP'];
			cumulatedFP += torsoPRF[torsoDivisions[l]]['FP'];
			cumulatedFN += torsoPRF[torsoDivisions[l]]['FN'];
			torsoOutputString3 = torsoOutputString3 + str(torsoDivisions[l]) + ';' 
			#prec = 1.0 * torsoPRF[torsoDivisions[l]]['TP'] / (torsoPRF[torsoDivisions[l]]['TP'] + torsoPRF[torsoDivisions[l]]['FP']) # Add P
			prec = 1.0 * cumulatedTP / (cumulatedTP + cumulatedFP) 
			#rec = 1.0 * torsoPRF[torsoDivisions[l]]['TP'] / (torsoPRF[torsoDivisions[l]]['TP'] + torsoPRF[torsoDivisions[l]]['FN']) # Add R
			rec = 1.0 * cumulatedTP / (cumulatedTP + cumulatedFN) 
			fm = 2.0 * prec * rec / (prec+rec)
			torsoOutputString2 += str( prec ) + ':' + str(rec) + ':' + str(fm) + ';\n'
		for f in range(NumFeatures):
			T_sum[f] = T_sum[f] + torsoFeatures[torsoDivisions[l]][f]
	print torsoOutputString3
	print torsoOutputString1
	print torsoOutputString2

	if torsoQuery_linked + torsoQuery_unlinked > 0 :
		print 'Total Torso queries - linked = %0.4f%%(%d) and unlinked = %0.4f%%(%d)' %(100.0*torsoQuery_linked/(torsoQuery_linked+torsoQuery_unlinked), torsoQuery_linked,\
		 100.0*torsoQuery_unlinked/(torsoQuery_linked+torsoQuery_unlinked), torsoQuery_unlinked)
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print 'Torso Avg Feature Values'
		T_AVG = [ 1.0*f/(torsoQuery_linked + torsoQuery_unlinked) for f in T_sum ]
		print T_AVG


	if tailQuery_linked+tailQuery_unlinked > 0 :
		print 'All Tail queries - linked = %0.4f%%(%d) and unlinked = %0.4f%%(%d)' %(100.0*(tailQuery_linked+torsoQuery_linked)/(tailQuery_linked+tailQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked),\
		(tailQuery_linked+torsoQuery_linked), 100.0* (tailQuery_unlinked+torsoQuery_unlinked)/(tailQuery_linked+tailQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked), (tailQuery_unlinked+torsoQuery_unlinked))
	print 'Total = %d, missingEntInlink = %d, missingQueryWikiName = %d, Tail_Threshold = %d' %((PopularQuery_linked+PopularQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked+tailQuery_linked+tailQuery_unlinked), len(missingEntInlink), missingQueryWikiName, Tail_Threshold)
	#print missingEntInlink
	#print 'Popularity in linking: Popular queries - linked = (%d) and unlinked = (%d), Tail queries - linked = (%d) and unlinked = (%d), Total = %d, missingEntInlink = %d, missingQueryWikiName = %d' %( PopularQuery_linked,  PopularQuery_unlinked,  torsoQuery_linked,  torsoQuery_unlinked, (PopularQuery_linked+PopularQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked), missingEntInlink, missingQueryWikiName)
	
	'''
	print 'RANK of linking for %d queries' %(TP)
	for rank in range(1,21):
		print 'Candidate @%d linked for %d queries' %(rank, Result[rank])

	print 'Queries with >1 links = %d' %(multiLinkedQuery)

	print '  true positive %0.4f%%' %(100.0 * TP / (TP+TN+FP+FN))
	if TP > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), TP = %d' %( 100.0 * TP_per/TP, TP_per, 100.0 * TP_gpe/TP, TP_gpe, 100.0 * TP_org/TP, TP_org, TP)
		AVG = [1.0*k/TP for k in avj_tp ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * TP_web/TP, 100.0 * TP_news/TP)
		
	print '  true negative %0.4f%%' %(100.0 * TN / (TP+TN+FP+FN))
	if TN > 0: 
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), TN = %d ' %(100.0 * TN_per/TN, TN_per, 100.0 * TN_gpe/TN, TN_gpe, 100.0 * TN_org/TN, TN_org, TN)
		AVG = [1.0*k/TN for k in avj_tn ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * TN_web/TN, 100.0 * TN_news/TN)

	print ' false positive %0.4f%%' %(100.0 * FP / (TP+TN+FP+FN))
	if FP > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), FP = %d ' %(100.0 * FP_per/FP, FP_per, 100.0 * FP_gpe/FP, FP_gpe, 100.0 * FP_org/FP, FP_org, FP)
		AVG = [1.0*k/FP for k in avj_fp ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * FP_web/FP, 100.0 *  FP_news/FP)

	print ' false negative %0.4f%%' %(100.0 * FN / (TP+TN+FP+FN))
	if FN > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), FN = %d ' %(100.0 * FN_per/FN, FN_per, 100.0 * FN_gpe/FN, FN_gpe, 100.0 * FN_org/FN, FN_org, FN)
		AVG = [1.0*k/FN for k in avj_fn ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * FN_web/FN, 100.0 * FN_news/FN)
	'''
	if datasetFlag:
		Pdoc = 0.0
		PdocCount = 0
		for qId in Pmacro:
			if Pmacro[qId]['TP'] > 0 :
				#print '%0.4f' %(1.0 * Pmacro[qId]['TP'] / (Pmacro[qId]['TP'] + Pmacro[qId]['FP']))
				Pdoc = Pdoc + 1.0 * Pmacro[qId]['TP'] / (Pmacro[qId]['TP'] + Pmacro[qId]['FP'])
				PdocCount = PdocCount + 1
		print 'AIDA Pmacro = %0.4f' %(Pdoc/PdocCount)
		
	#Uncomment this to get a report on console
	'''
	if reportParameter == 'TP':
		print " \n True Positive Report "
		for i in range(TP):
			print TP_list[i], TP_track[i]
	elif reportParameter == 'FP':			
		print "\n False Positive Report "
		for i in range(FP):
			print FP_list[i], FP_track[i]
	elif reportParameter == 'TN':			
		print "\n True Negative Report "
		for i in range(TN):
			print TN_list[i], TN_track[i]
	elif reportParameter == 'FN':
		print "\n False Negative Report"
		for i in range(FN):
			print FN_list[i], FN_track[i]
	'''
	return (TP_per, TP_gpe, TP_org, TN, FP, FN)

	#EA end

def classify_nolink(train_set,test_set,datasetFlag):
	global entInlinkMap, CommonSenseCand, CommonSenseCandMissed
	#Classification algorithm.
	print 'Train set :'
	train_feat,train_target,track_train=numpyCreate_EA(train_set);#,12);#train_feat,train_target=numpyCreate(train_set);
	print 'Test set :'
	#test_feat,test_target=numpyCreate(test_set);
	test_feat,test_target,track_test=numpyCreate_EA(test_set);#,12);#test_feat,test_target=numpyCreate2(test_set,5); #TAC

	#Build the pipeline
	clf=Pipeline([('clf', RandomForestClassifier(max_features=3,n_estimators=100))]);#, verbose=3))]);
	#clf=Pipeline([('clf', ExtraTreesClassifier(max_features=3,n_estimators=100))]);
	#clf=Pipeline([('clf', AdaBoostClassifier(n_estimators=100))]);0
	#clf=Pipeline([('clf', GradientBoostingClassifier(n_estimators=10000, learning_rate=0.02, max_depth=4, random_state=0, verbose=3))]);

	#Training
	clf.fit(train_feat,train_target);

	#Testing
	predicted=clf.predict(test_feat);

	#print np.mean(predicted==test_target);	

	target_names=['0','1'];
	#print(metrics.confusion_matrix(test_target, predicted));
	print(metrics.classification_report(test_target, predicted,target_names=target_names));
	#print (metrics.precision_score(test_target, predicted, labels=None, pos_label='1', average='micro'))
	#print (metrics.precision_score(test_target, predicted, labels=None, pos_label='1', average='binary'))#macro'))
	#print(metrics.precision_recall_fscore_support(test_target, predicted,labels=None, pos_label='1', average=None));

	#EA start
	#datasetFlag = False #AIDA - True, TAC - False
	populateELmaps(datasetFlag)
	test_feature_set = getFeatures(test_set)
	TP, TN, FP, FN = 0,0,0,0
	#F1, F2, F3, F4, F5, F6, F7, F8, F9 = 0,0,0,0,0,0,0,0,0
	avj_tp = [0.0] * NumFeatures #instantiate average list with 0.0
	avj_tn = [0.0] * NumFeatures
	avj_fp = [0.0] * NumFeatures
	avj_fn = [0.0] * NumFeatures
	Pmacro = {} # query to TP,TN,FP,FN counts map #AIDA
	PosNegCounts = {'TP':0,'TN':0,'FP':0,'FN':0} 

	TP_gpe, TP_per, TP_org, TN_gpe, TN_per, TN_org, FP_gpe, FP_per, FP_org, FN_gpe, FN_org, FN_per = 0,0,0,0, 0,0,0,0, 0,0,0,0
	TP_web, TP_news, TN_web, TN_news, FP_web, FP_news, FN_web, FN_news = 0,0,0,0, 0,0,0,0
	missingTypeCount = 0
	queryResultsMap = {}

	qId_stored = ''
	candRank = 0
	candLinkingMap = {} #map of candRank to linking status
	
	for i in range(len(track_test)):
		if datasetFlag : #AIDA
			goldTypeKey = track_test[i].split(':')[1]#AIDA
			qId = 'Q'+track_test[i].split(':')[0]
		else :
			goldTypeKey = track_test[i].split(':')[0]#TAC
			qId = track_test[i].split(':')[0]

		if qId_stored != qId : #next query. Print out last qId
			queryResultsMap[qId_stored] = candLinkingMap
			qId_stored = qId
			candRank = 0
			candLinkingMap = {}

		if CommonRare:
			candId = track_test[i].split(':')[1]
			if candId in querySenseMap[qId]:
				if querySenseMap[qId][candId] > CommonRareSenseThreshold:
					CommonSenseCand = CommonSenseCand + 1 # Cand links to its common sense
					#print ' Cand %s links to Common sense' %(candId)
			else:
				CommonSenseCandMissed = CommonSenseCandMissed + 1


		candRank = 	candRank + 1
		if predicted[i]==test_target[i]:
			if predicted[i] == '1':
				candLinkingMap[candRank] = 1
			else:
				candLinkingMap[candRank] = 0				
		else:
			candLinkingMap[candRank] = 0

		qString = elQueries[qId] 
		if datasetFlag : #AIDA. Pmacro
			if qId in Pmacro:
				PosNegCounts = Pmacro[qId]
			else:
				PosNegCounts = {'TP':0,'TN':0,'FP':0,'FN':0} 

			if predicted[i]==test_target[i]:
				if predicted[i] == '1':			#TP = TP + 1
					PosNegCounts['TP'] = PosNegCounts['TP'] + 1
				else: #TN = TN + 1
					PosNegCounts['TN'] = PosNegCounts['TN'] + 1
			else:
				if predicted[i] == '1': #FP = FP + 1
					PosNegCounts['FP'] = PosNegCounts['FP'] + 1
				else: #FN = FN + 1 
					PosNegCounts['FN'] = PosNegCounts['FN'] + 1
			Pmacro[qId] = PosNegCounts

		else: #TAC Genre of query text
			if "docId:eng-" in qString: #web text
				QueryTextGenre = 'WEB'
			elif "_ENG_" in qString : # English News Text. Could be New York Times, Associated Press, Xinhua News 
				QueryTextGenre = 'NEWS'
		
		if goldTypeKey in elTypes: #if type info exists for this 
			goldType = elTypes[goldTypeKey]
			if predicted[i]==test_target[i]:
				if predicted[i] == '1':			#TP = TP + 1
					if goldType == 'ORG':
						TP_org = TP_org + 1
					elif goldType == 'PER':
						TP_per = TP_per + 1
					elif goldType == 'GPE':
						TP_gpe = TP_gpe + 1
					#Average feature values
					for j in range(len(test_feature_set[i])-1):
						avj_tp[j] = avj_tp[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							TP_web = TP_web + 1
						elif QueryTextGenre == 'NEWS':
							TP_news = TP_news + 1
						
				else: #TN = TN + 1
					if goldType == 'ORG':
						TN_org = TN_org + 1
					elif goldType == 'PER':
						TN_per = TN_per + 1
					elif goldType == 'GPE':
						TN_gpe = TN_gpe + 1
					#Average feature values
					for j in range(len(test_feature_set[i])-1):
						avj_tn[j] = avj_tn[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							TN_web = TN_web + 1
						elif QueryTextGenre == 'NEWS':
							TN_news = TN_news + 1
			else:
				if predicted[i] == '1': #Negative predicted as positive #FP = FP + 1				
					if goldType == 'ORG':
						FP_org = FP_org + 1
					elif goldType == 'PER':
						FP_per = FP_per + 1
					elif goldType == 'GPE':
						FP_gpe = FP_gpe + 1

					for j in range(len(test_feature_set[i])-1):
						avj_fp[j] = avj_fp[j] + test_feature_set[i][j]
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							FP_web = FP_web + 1
						elif QueryTextGenre == 'NEWS':
							FP_news = FP_news + 1
					'''
					print ' %s , %s %s => %s : %s' %(track_test[i], QueryTextGenre, goldType ,predicted[i],test_target[i])
					print test_feature_set[i]
					if qId in elContexts:
						print elContexts[qId]					
					print qString
					if datasetFlag :
						print ' CAND : %s\n ' %( elCandidate[goldTypeKey])
					else :
						print ' CAND : %s\n ' %(createTrainData.getTitle(int(track_test[i].split(':')[1])))
					'''

				else: #FN = FN + 1 #positive predicted as negative
					if goldType == 'ORG':
						FN_org = FN_org + 1
					elif goldType == 'PER':
						FN_per = FN_per + 1
					elif goldType == 'GPE':
						FN_gpe = FN_gpe + 1

					for j in range(len(test_feature_set[i])-1):
						avj_fn[j] = avj_fn[j] + test_feature_set[i][j]						
					if not datasetFlag : 
						if QueryTextGenre == 'WEB':
							FN_web = FN_web + 1
						elif QueryTextGenre == 'NEWS':
							FN_news = FN_news + 1
					'''
					print ' %s , %s %s => %s : %s' %(track_test[i], QueryTextGenre, goldType ,predicted[i],test_target[i])
					print test_feature_set[i]
					if qId in elContexts:
						print elContexts[qId]
					print qString
					if datasetFlag :
						print ' CAND : %s\n ' %( elCandidate[goldTypeKey])
					else :
						print ' CAND : %s\n ' %(createTrainData.getTitle(int(track_test[i].split(':')[1])))
					'''
		else :
			missingTypeCount = missingTypeCount + 1

	print 'Missing cand type for %d' %(missingTypeCount)
	TP = TP_per + TP_gpe + TP_org
	TN = TN_per + TN_gpe + TN_org
	FP = FP_per + FP_gpe + FP_org
	FN = FN_per + FN_gpe + FN_org


	#per query analysis
	rank = 1
	Result = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0}
	multiLinkedQuery,torsoQuery_linked, torsoQuery_unlinked, PopularQuery_linked, PopularQuery_unlinked , missingQueryWikiName, missingEntInlink = 0,0,0,0,0,0,0
	print '%d query results populated' %(len(queryResultsMap))
	for query in queryResultsMap:
		#print queryResultsMap[query]
		for rank in range(1,21):
			if len(queryResultsMap[query]) >= rank:
				if queryResultsMap[query][rank] == 1 :
					Result[rank] = Result[rank] + 1

		#sanity check- Does a query have multiple links
		#if len([k for k in range(1,len(queryResultsMap[query])) if queryResultsMap[query][k]==1]) > 1 :
		#	multiLinkedQuery = multiLinkedQuery  + 1 #empirically found out as 0 in TAC and positive in AIDA
			#Popular versus tail queries
		
		if query in elEntities:
			#print 'query = %s, enWikiname = %s' %(query, elEntities[query])
			#print "===================================="
			#print entInlinkMap
			#print "===================================="
			if elEntities[query] in entInlinkMap:
				#print 'query = %s, inlinks = %s' %(query, entInlinkMap[elEntities[query]])
				if int(entInlinkMap[elEntities[query]]) < Tail_Threshold :
					if len([k for k in range(1,len(queryResultsMap[query])) if queryResultsMap[query][k]==1]) >= 1 :
						torsoQuery_linked = torsoQuery_linked + 1
					else:
						torsoQuery_unlinked = torsoQuery_unlinked + 1
				else:
					if len([k for k in range(1,len(queryResultsMap[query])) if queryResultsMap[query][k]==1]) >= 1 :
						PopularQuery_linked = PopularQuery_linked + 1
					else:
						PopularQuery_unlinked = PopularQuery_unlinked + 1
			else:
				missingEntInlink = missingEntInlink + 1
		else:
			missingQueryWikiName = missingQueryWikiName + 1


	#Common and Rare Senses
	if CommonRare : #Common and Rare Sense candidates
		print ' Query division by Common and Rare senses : Common sense = %d, Rare sense = %d, CommonRareSenseThreshold = %0.4f' %(CommonSenseQuery, len(querySenseMap) - CommonSenseQuery, CommonRareSenseThreshold)
		print ' Candidate division by Common and Rare senses : Common sense = %d, Rare sense = %d, CommonSenseCandMissed = %d' %(CommonSenseCand, len(track_test) - CommonSenseCand, CommonSenseCandMissed)

	print 'Popularity in linking: Popular queries - linked = %0.4f%%(%d) and unlinked = %0.4f%%(%d), Tail queries - linked = %0.4f%%(%d) and unlinked = %0.4f%%(%d), Total = %d, missingEntInlink = %d, missingQueryWikiName = %d, Tail_Threshold = %d' %(100.0*PopularQuery_linked/(PopularQuery_linked+PopularQuery_unlinked), PopularQuery_linked, 100.0*PopularQuery_unlinked/(PopularQuery_linked+PopularQuery_unlinked), PopularQuery_unlinked, 100.0*torsoQuery_linked/(torsoQuery_linked+torsoQuery_unlinked), torsoQuery_linked, 100.0*torsoQuery_unlinked/(torsoQuery_linked+torsoQuery_unlinked), torsoQuery_unlinked, (PopularQuery_linked+PopularQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked), missingEntInlink, missingQueryWikiName, Tail_Threshold)
	#print 'Popularity in linking: Popular queries - linked = (%d) and unlinked = (%d), Tail queries - linked = (%d) and unlinked = (%d), Total = %d, missingEntInlink = %d, missingQueryWikiName = %d' %( PopularQuery_linked,  PopularQuery_unlinked,  torsoQuery_linked,  torsoQuery_unlinked, (PopularQuery_linked+PopularQuery_unlinked+torsoQuery_linked+torsoQuery_unlinked), missingEntInlink, missingQueryWikiName)

	print 'RANK of linking for %d queries' %(TP)
	for rank in range(1,21):
		print 'Candidate @%d linked for %d queries' %(rank, Result[rank])

	print 'Queries with >1 links = %d' %(multiLinkedQuery)

	print '  true positive %0.4f%%' %(100.0 * TP / (TP+TN+FP+FN))
	if TP > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), TP = %d' %( 100.0 * TP_per/TP, TP_per, 100.0 * TP_gpe/TP, TP_gpe, 100.0 * TP_org/TP, TP_org, TP)
		AVG = [1.0*k/TP for k in avj_tp ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * TP_web/TP, 100.0 * TP_news/TP)
		
	print '  true negative %0.4f%%' %(100.0 * TN / (TP+TN+FP+FN))
	if TN > 0: 
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), TN = %d ' %(100.0 * TN_per/TN, TN_per, 100.0 * TN_gpe/TN, TN_gpe, 100.0 * TN_org/TN, TN_org, TN)
		AVG = [1.0*k/TN for k in avj_tn ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * TN_web/TN, 100.0 * TN_news/TN)

	print ' false positive %0.4f%%' %(100.0 * FP / (TP+TN+FP+FN))
	if FP > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), FP = %d ' %(100.0 * FP_per/FP, FP_per, 100.0 * FP_gpe/FP, FP_gpe, 100.0 * FP_org/FP, FP_org, FP)
		AVG = [1.0*k/FP for k in avj_fp ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * FP_web/FP, 100.0 *  FP_news/FP)

	print ' false negative %0.4f%%' %(100.0 * FN / (TP+TN+FP+FN))
	if FN > 0:
		print 'PER = %0.4f%% (%d), GPE = %0.4f%% (%d), ORG = %0.4f%% (%d), FN = %d ' %(100.0 * FN_per/FN, FN_per, 100.0 * FN_gpe/FN, FN_gpe, 100.0 * FN_org/FN, FN_org, FN)
		AVG = [1.0*k/FN for k in avj_fn ]
		#print 'Average Feature Values '#F1 %0.4f, F2 %0.4f, F3 %0.4f, F4%0.4f, F5%0.4f, F6%0.4f, F7%0.4f, F8%0.4f, F9%0.4f' AVG
		print AVG
		if not datasetFlag : 
			print 'WEB = %0.4f%%, NEWS = %0.4f%%' %(100.0 * FN_web/FN, 100.0 * FN_news/FN)

	if datasetFlag:
		Pdoc = 0.0
		PdocCount = 0
		for qId in Pmacro:
			if Pmacro[qId]['TP'] > 0 :
				#print '%0.4f' %(1.0 * Pmacro[qId]['TP'] / (Pmacro[qId]['TP'] + Pmacro[qId]['FP']))
				Pdoc = Pdoc + 1.0 * Pmacro[qId]['TP'] / (Pmacro[qId]['TP'] + Pmacro[qId]['FP'])
				PdocCount = PdocCount + 1
		print 'AIDA Pmacro = %0.4f' %(Pdoc/PdocCount)

	return (TP_per, TP_gpe, TP_org, TN, FP, FN)
	#EA end

#true for EA, false for unlink
def UniFile(EAflag): #Train and test samples in single file
	#Artificial python main.
	if len(sys.argv)!=3:
		print("Missing arguments.\nFORMAT : python classify.py <file> <numfolds>");
		sys.exit();
	
	#Read the entire dataset from the specified file.
	file=sys.argv[1];
	dataset=[];
	with open(file) as f:
		dataset=f.readlines();

	#Find the dataset
	datasetFlag = False
	if 'AIDA' in file:
		datasetFlag = True

	if not EAflag:
		readNolinkPairs('noInlinkQCmap.txt')

	#Do n fold cross folding.
	num_folds=int(sys.argv[2]);
	subset_size=len(dataset)/num_folds;
	TP_per, TP_gpe, TP_org, TN, FP, FN = 0,0,0,0,0,0
	for i in range(num_folds):
		test_set=dataset[i*subset_size:][:subset_size];
		train_set=dataset[:i*subset_size]+dataset[(i+1)*subset_size:];
		if EAflag:
			TP_perfold, TP_gpefold, TP_orgfold, TNfold, FPfold, FNfold = classify_EA(train_set,test_set,datasetFlag);
			TP_per = TP_per + TP_perfold
			TP_gpe = TP_gpe + TP_gpefold
			TP_org = TP_org + TP_orgfold
			TN = TN + TNfold
			FP = FP + FPfold
		else: #un-inlinked
			classify_nolink(train_dataset,test_dataset, datasetFlag)

	if EAflag:
		TP = TP_per + TP_gpe + TP_org
		print 'After %d fold validation, TP_per = %d, TP_gpe = %d, TP_org = %d, TN = %d, FP= %d, FN = %d '%(num_folds, TP_per, TP_gpe, TP_org, TN, FP, FN)
		print 'P = %0.4f R = %0.f'%(1.0*TP/(TP+FP), 1.0*TP/(TP+FN))

def BiFile(EAflag): #Train and test samples in two( different) file
	if len(sys.argv)!=3:
		print("Missing arguments.\nFORMAT : python classify.py <Train_file> <Test_file>");
		sys.exit();
	
	#Read the train file.
	train_file=sys.argv[1];
	train_dataset=[];
	with open(train_file) as f:
		train_dataset=f.readlines();
	datasetFlag = False
	if 'AIDA' in train_file:
		datasetFlag = True
	
	#Read the test file.
	test_file=sys.argv[2];
	test_dataset=[];
	with open(test_file) as f:
		test_dataset=f.readlines();

	#test_feature_set = getFeatures(test_dataset) ; print("features set length = %d" %(len(test_feature_set)))
	if EAflag :
		classify_EA(train_dataset,test_dataset, datasetFlag);
	else: #unlinked
		readNolinkPairs('noInlinkQCmap.txt')
		classify_nolink(train_dataset,test_dataset, datasetFlag)
	
#EA start
#populateELmaps(False)
#Assuming TAC
#if not datasetFlag:
if CommonRare :
	fc= open(TACsourceCandTest, 'r')
	populateSenseMaps(fc, False)  
	fc.close()
#EA stop

BiFile(True); 
#UniFile(True)
#analyzeFeatures()