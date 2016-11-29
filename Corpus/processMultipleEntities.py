#import python packages
import json
import urllib2
import os
import sys
import string
import nltk
from nltk import word_tokenize, pos_tag
from multiprocessing import Process, Queue
import time
# from boilerpipe.extract import Extractor
import pymongo
from pymongo import MongoClient
import mongodbClass as mdb
import cPickle as pickle
########################################################################################
# Enter primary entity name. Give supporting strings and/or freebase id.
########################################################################################
from HtmlTextExtraction import getLinks_api_search,getLinks_cmu_search, getLinks_custom_search, extractDataFromLink
from getEntities import collectEntities
from clusterAndNormalise_glove_par import entityClusterAndNormalise
from dereferenceOllieOutput import ReplaceCorefPointers
from ent_linking_nn import inference_test
from WebScraping import webScraping


sys.setrecursionlimit(10000);

def Main(primary_ent,query_strings,primary_ent_type, extractTriples):
    nellMapObj = mdb.mongodbDatabase('nell_mapped_triples_collection')
    nellMapCol = nellMapObj.docCollection
    oldTriples = nellMapCol.find_one({'primaryEnt':primary_ent})
    #if oldTriples == None:
    #    webScraping(primary_ent, query_strings)
    #    entityClusterAndNormalise(primary_ent,primary_ent_type)
        #success = inference_test(primary_ent,primary_ent_type)
    webScraping(primary_ent, query_strings, extractTriples)
    if(extractTriples):
        entityClusterAndNormalise(primary_ent,primary_ent_type)
    nellMapObj.client.close()

###########################################
## Starting point for running entice in batch mode
########################################### 

'''fi = open("Primary Entities", "r")
for line in fi:
    data = line.split('	')
    primary_ent = data[0]
    query_strings = [data[2]]
    primary_ent_type = data[1]
    extractTriples = False
    print "Extracting triples for ", primary_ent;
    Main(primary_ent,query_strings,primary_ent_type, extractTriples)

primary_ent = 'Suresh Raina'; 
query_strings = ['cricketer'];
primary_ent_type = 'Person';
extractTriples = True;
Main(primary_ent, query_strings, primary_ent_type, extractTriples);'''

def getAllEntities():
    print("Training anchors for entities in TAC and AIDA...")
    #.1. collect entities dealt with in evaluation
    trainingEntities = {}

    #reading from EDqueries.csv
    with open('./Priya/EDqueries.csv', 'r') as f1:
        for line in  f1.readlines():
            for entitypair in line.split(':::'):
                for entity in entitypair.split("#"):
                    if entity not in trainingEntities: 
                        trainingEntities[entity] = 0;
    print 'Found %d entities from EDqueries.csv '%(len(trainingEntities))

    #reading from evalEntitiesMap.txt 
    with open('./Priya/evalEntitiesMap.txt', "r") as f2:
        for entity in f2.readlines():
            if entity not in trainingEntities:
                trainingEntities[entity] = 0
    print 'Read %d entities of evaluation.'%len(trainingEntities)
    return trainingEntities;

trainingEntities = getAllEntities();

dfile = {};
for file in os.listdir("./Priya/Sentences_new"):
    if file.endswith(".txt"):
        fn = file.strip('.txt');
        #filename = fn.strip("\n.txt")
        #os.rename(file, filename + '.txt')
        #count += 1;
        dfile[fn] = 1;

for file in os.listdir("./Priya/Sentences"):
    if file.endswith(".txt"):
        fn = file.strip('.txt');
        #filename = fn.strip("\n.txt")
        #os.rename(file, filename + '.txt')
        #count += 1;
        dfile[fn] = 1;

for file in os.listdir("./Priya/Sentences_new1"):
    if file.endswith(".txt"):
        fn = file.strip('.txt');
        #filename = fn.strip("\n.txt")
        #os.rename(file, filename + '.txt')
        #count += 1;
        dfile[fn] = 1;

for file in os.listdir("./Priya/Sentences_new2"):
    if file.endswith(".txt"):
        fn = file.strip('.txt');
        #filename = fn.strip("\n.txt")
        #os.rename(file, filename + '.txt')
        #count += 1;
        dfile[fn] = 1;

list = dfile.keys();
for i in list:
    for k in trainingEntities.keys():
        kk = k.replace('/', '_')
        if ((kk in i) or (i in kk)) and abs(len(k) - len(i)) <= 2:
            #count += 1;
            #print k, i, len(k), len(i);
            del trainingEntities[k];
            break;

dfile1 = {};
for k in trainingEntities.keys():
    kk = k.replace('/', '_');
    done = 0;
    '''
    for file in os.listdir("./Priya/Sentences_new1"):
        if file.endswith(".txt"):
            fn = file.strip('.txt');
            dfile1[fn] = 1;
    for i in dfile1.keys():
        if(kk in i) or (i in kk):
            done = 1;
            break;
    '''
    if done == 0:
        primary_ent = k.strip();
        query_strings = []
        primary_ent_type = k
        extractTriples = False
        print 'Extracting triples for ', primary_ent;
        webScraping(primary_ent.replace('/', '_'), query_strings, extractTriples)
