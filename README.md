<h1>ELDEN: Improved Entity Linking using Densified Knowledge Graphs</h1>
This software is the implementation of the paper "ELDEN: Improved Entity Linking using Densified Knowledge Graphs" to be presented at 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT 2018) at New Orleans, Louisiana, June 1 to June 6, 2018. 

<h2>Requirements</h2>

Code is written in Python (2.7), Torch and Lua (Luajit)
Using the pre-trained word2vec vectors from gensim will require downloading it from https://radimrehurek.com/gensim/models/word2vec.html
Co-occurance matrix and other datafiles can be downloaded at https://www.dropbox.com/s/wqduqde7pv8cr76/ELDEN_Corpus.tar.gz?dl=0

<h2>Running the models</h2>
This package contains the four steps (folders A to D) of implementation, followed by Evaluation. We suggest running the system in this order.

A. Corpus :
1. Wikipedia (clean as specified in paper) 
2. Web Corpus = trainingEntities.py, processMultipleEntities.py, WebScraping.py

B. Dataset :
1. TAC2010 = TACforNED
2. CoNLL = https://github.com/masha-p/PPRforNED 
Please cite the respective papers when using these datasets.

C. Preprocess:
1. Create entity co-location index. 
     python2.7 pmi_index.py base_co.npy/None vocab.pickle output_file file_scraped_from_web
2. Start PMI Server. 
     python pmi_service.py
3. Train entity embeddings. 
     th> main.lua <<word2vec.lua>>
4. Start Embedding Distance Servers. 
     th> EDServer.lua

D. Entity Linker:
1. Create train and test dataset
     python createTrainData.py
2. Run Entity Linker
     python classify.py

E. Evaluation :
1. Head entities versus tail entities statistics
     python TailEntities.py

Kindly cite the paper if you are using the software 




