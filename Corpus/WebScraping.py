#import python packages
import json
import urllib2
import os
import string
import nltk
from nltk import word_tokenize, pos_tag
from multiprocessing import Process, Queue
import time
#######
# from local files
import mongodbClass as mdb
from HtmlTextExtraction import getLinks_api_search,getLinks_cmu_search, getLinks_custom_search, extractDataFromLink, extractSentencesFromLink
from dexter_ent_linking import link_entities
##
#read Keys
##
keys = {}
execfile("config/keys.txt", keys)
api_key = keys["api_key"]
cx_key = keys["cx_key"]

TIMEOUT = 120
ent_search_wiki_url = ''
reject = True
####
# entity - mid of the entity
# entType - entity type (freebase defined)
# returns attribute values of the specified enity
####
def getAttribValues(entity, entType):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{  '*': None, 'mid': entity,  'type':entType }]
    params = {
            'query': json.dumps(query),
            'key': api_key
    }
    try:
        urls = service_url + '?' + urllib.urlencode(params)
        extractor = Extractor(extractor='ArticleExtractor', url=urls)
        extracted_text = extractor.getText()
        response = json.loads(extracted_text)
        return response
    except:
        #print "$$$$ERROR$$$"
        return None

##
# return the type json object.
# json object contains list of types defined for given entity
##
def getTypeListFromFreebase(entId):
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{  'mid': entId, 'name': [],  'type':[] }]

    params = {
            'query': json.dumps(query),
            'key': api_key
    }
    try:
        urls = service_url + '?' + urllib.urlencode(params)
        extractor = Extractor(extractor='ArticleExtractor', url=urls)
        extracted_text = extractor.getText()
        response = json.loads(extracted_text)
        return response
    except:
        return None

url_set = set()
def validLink(link):
    ''' logic to exclude bad links'''
    global url_set
    if "www.youtube.com" in link:
        return False
    if 'https://' in link:
        link = link.replace('https://','')
    if 'http://' in link:
        link = link.replace('http://','')
    if link in url_set:
        return False
    if reject and ('en.wikipedia.org' in link) and (ent_search_wiki_url.replace('http://','') != link):
        print "###### rejected ",link
        return False
    else:
        url_set.add(link)
        return True

###############
# @args input
# entToSearch - primary entity to search for
# mid - freebase machine id of the entity
# queryStrings - supporting strings about primary entity. This is to improve search result.
###############
def webScraping(entToSearch, queryStrings, extractTriples):
    global ent_search_wiki_url
    global reject
    entTypeList = []

    tempSet = set()
    valuesToSearch = set()
    valuesToSearch.add(entToSearch)
    for q in queryStrings:
        valuesToSearch.add(q)

    # to get the wikipedia url of the primary entity.
    # reject other wikipedia urls
    wikiPageTitleObj = mdb.mongodbDatabase('wikiPageTitle','wikiPageDB')
    wikiPageTitle_col = wikiPageTitleObj.docCollection

    link_str = entToSearch + ' ' + ' '.join(queryStrings)
    wiki_url = 'http://en.wikipedia.org/wiki/'
    link_ent_dict = link_entities(link_str.lower(),wikiPageTitle_col)
    ent_title = link_ent_dict[entToSearch.lower()]
    
    wikiPageTitleObj.client.close()
    
    if len(ent_title) != 0:
        ent_search_wiki_url = wiki_url + ent_title
        print "wiki url of primary entity ******",ent_search_wiki_url,"*********"
    else:
        print "linked dict",link_ent_dict,"*******"
        reject = False
    #if len(valuesToSearch) == 0:
        #valuesToSearch.add(entToSearch)

    fileCount = 1

    processList = []
    q = Queue()
    fileCount = 1;
    link_set = set()
    for qstr in valuesToSearch:
        if qstr == entToSearch:
            searchString = "\""+entToSearch+"\""
        else:
            qstr = qstr.strip('\n')
            searchString = entToSearch + " " + qstr
        start_time = time.time()
        linksList_api = getLinks_api_search(searchString,2)
        print("--- %s api ---" % (time.time() - start_time))
        start_time = time.time()
        linksList_cstm = getLinks_custom_search(searchString)   #last int to control the number of links
        print("--- %s custom ---" % (time.time() - start_time))
        start_time = time.time()
        linksList_cmu = getLinks_cmu_search(searchString)
        print("--- %s cmu ---" % (time.time() - start_time))
        #print "reminder--cmu search disabled"

        if linksList_api != None:
            for l in linksList_api:
                l = l.strip(' ')
                l = l.strip('\n')
                link_set.add(l)

        if linksList_cstm != None:
            print "kg ",len(linksList_cstm)
            for l in linksList_cstm:
                l = l.strip(' ')
                l = l.strip('\n')
                link_set.add(l)

        if linksList_cmu != None:
            print "cmu ",len(linksList_cmu)
            for l in linksList_cmu:
                l = l.strip(' ')
                l = l.strip('\n')
                link_set.add(l)

        print "link count :",len(link_set)
        # print link_set
    
    if link_set != None:
        for link in link_set:
            if validLink(link):
                print "^^^ added",link
                if extractTriples:
                    newProc = Process(target=extractDataFromLink, args=[q, link, entToSearch,fileCount, extractTriples])# call a function to do corenlp->sentcreate->ollie
                else:
                    newProc = Process(target=extractSentencesFromLink, args=[q, link, entToSearch,fileCount, extractTriples])# call a function to do corenlp->sentcreate->ollie
                fileCount += 1;
                processList.append(newProc)
                newProc.start()
    
    start = time.time()
    while time.time() - start <= TIMEOUT:
        if any(p.is_alive() for p in processList):
            time.sleep(1)  # Just to avoid hogging the CPU
        else:
            # All the processes are done, break now.
            break
    else:
        # We only enter this if we didn't 'break' above.
        print("timed out, killing all processes")
        for p in processList:
            p.terminate()
