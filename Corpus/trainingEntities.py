import os
trainingEntities={}

#TAC and AIDA dataset entities in EDqueries.csv and evalEntitiesMap.txt
def train_entitys():
    print("Training anchors for entities in TAC and AIDA...")
    #.1. collect entities dealt with in evaluation
    global trainingEntities
    
    #reading from EDqueries.csv
    with open('EDqueries.csv', 'r') as f1:
        for line in  f1.readlines():
            for entitypair in line.split(':::'):
                for entity in entitypair.split("#"):
                    if entity.strip() not in trainingEntities: 
                        trainingEntities[entity.strip()] = 0;
    print 'Found %d entities from EDqueries.csv '%(len(trainingEntities))

    #reading from evalEntitiesMap.txt 
    with open('evalEntitiesMap.txt', "r") as f2:
        for entity in f2.readlines():
            if entity.strip() not in trainingEntities:
                trainingEntities[entity.strip()] = 0
    print 'Read %d entities of evaluation.'%len(trainingEntities)

def getEntities(inputFolder):
    missingEntities = []
    counter = 0
    for f in os.listdir(inputFolder):
        counter += 1
        #print 'File : ', f        
        if f.endswith(".txt"):
            f = f.strip('.txt');
        elif f.endswith("\n"):
            f = f.strip('\n');
        f=f.strip()
        if f not in trainingEntities:
            #print f
            missingEntities.append(f)
    print inputFolder,' has ', str(counter), ' files, ',str(len(missingEntities)),' missing entities'
    return missingEntities


def missCount():
    dfile = {};
    for file in os.listdir("./Sentences_new"):
        if file.endswith(".txt"):
            fn = file.strip('.txt');
            #filename = fn.strip("\n.txt")
            #os.rename(file, filename + '.txt')
            #count += 1;
            dfile[fn] = 1;

    count = 0;
    list = dfile.keys();
    for i in list:
        for k in trainingEntities.keys():
            if ((k in i) or (i in k)) and abs(len(k) - len(i)) <= 2:
                count += 1;
                print k, i, len(k), len(i);
                break;

    print count;

#main
train_entitys()
missingEntities = getEntities('./Sentences')