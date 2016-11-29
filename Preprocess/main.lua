--[[
Config file for skipgram with negative sampling 
original source - https://github.com/yoonkim/word2vec_torch
modified - April 2016 by priya.r
--]]
--CPU version - July 20

require("sys")
require("io")
require("os")
require("paths")
require("torch")
dofile("word2vec.lua")

local threads = require 'threads'

startTime = sys.clock()

-- Default configuration
config = {}
config.corpus = "corpus.txt" -- input data folder
config.vocab = "wikiVocab.csv" --vocab file created using vocabMake.py
config.weight = "weightFile2" --word weigths( embeddings) stored here
config.pretrain =  "gensimWord2vecModel.csv" --word embeddings created by gensim
config.window = 5 -- (maximum) window size
config.dim = 100 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e4 --originaly 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 10 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 3 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first
config.entityTrain = "EntityTrainingSamples.csv_full"
config.slice = 1 --slice of EntityTrainingSamples.csv_slice

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-vocab", config.vocab)
cmd:option("-pretrain", config.pretrain)
--cmd.option("-weight_file", config.weight)
cmd:option("-window", config.window)
cmd:option("-minfreq", config.minfreq)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-min_lr", config.min_lr)
cmd:option("-neg_samples", config.neg_samples)
cmd:option("-table_size", config.table_size)
cmd:option("-epochs", config.epochs)
cmd:option("-gpu", config.gpu)
cmd:option("-stream", config.stream)
cmd:option("-entityTrain", config.entityTrain)
cmd:option("-slice", config.slice)

params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end
--[[for i,j in pairs(config) do
    print(i..": "..j)
end
]]--

if(config.gpu>0) then
    cutorch.setDevice(config.gpu)
end



--config.pool = pool

m = Word2Vec(config)
	-- Train model
	--[[m:build_vocab(config.vocab)
	m:build_table()
	m:model_erect()

	for k = 1, config.epochs do
	    m.lr = config.lr -- reset learning rate at each epoch
	    m:train_model(config.corpus)
	    m:store_wordEmbeddings()
	end

	m:load_weights() --weightFile2 has weights after batchProcessing
	]]--

	--Train from pretrained word vectors from gensim
--m:create_gensim_weights_entities()


--[[m:load_weights_entities(true)
m:model_erect()
m:build_table()

local word1 = 'Michelle_Obama'
local word2 = 'Barack_Obama'

--local sim = m:getSimilarity(word1, word2)
--if sim ~= nil then 
--	print (string.format(' Initial %s : %s = %0.6f', word1, word2, sim[1]))
--end

m:train_entities_for_entity(word1) --m:train_anchor_for_entity(word1)
m:train_entities_for_entity(word2)--m:train_anchor_for_entity(word2)
]]--

--[[
sim = m:getSimilarity(word1, word2)
if sim ~= nil then 
	print (string.format(' Final %s : %s = %0.6f', word1, word2, sim[1]))
end
]]--

--create EDwlm start
--[[m:load_gensim_weights_entities(true)
m:model_erect()
		--Update the model with new training documents
		--m:train_model(config.corpus) -- words in files in directory config.corpus will be added to model

		--print(string.format('Pgid = %d',m:getPageId("Indian_Institute_of_Science"))) --Michelle_Obama")))
m:build_table()
		--train the model with entity-anchor_text evidences
		--m:train_anchor()

		--train the model with entity-entity evidences
	--m:train_entity()
	--m:train_anchor_entity()
	--m:store_train_entity_weight(config.slice) --mandatory call after m:train_entity()
m:train_entity_eval()
]]--
--create EDwlm end


--create EDpmi start
m:load_gensim_weights_entities(true)
m:model_erect()
m:build_table()
print (string.format(' Loading took %.2f seconds',sys.clock() - startTime ))
--m:train_entities_for_entity_TR('Barack_Obama') 
m:train_entity_eval()
--create EDpmi end

-------------------Training entity atop anchors-----------
--[[m:load_weights_entities(true)
m:model_erect()
m:build_table()

local word1 = 'Algeria'
local word2 = 'Algiers'

local sim = m:getSimilarity(word1, word2, true)
if sim ~= nil then 
	print (string.format(' Initially %s : %s = %0.6f', word1, word2, sim[1]))
end

for epoch=1,config.epochs do
	m:train_anchor_for_entity(word1) --'Audi')--Michelle_Obama')
	m:train_anchor_for_entity(word2)
end

sim = m:getSimilarity(word1, word2, true)
if sim ~= nil then 
	print (string.format(' Final %s : %s = %0.6f', word1, word2, sim[1]))
end
]]--working

--eval
--[[
m:load_weights_entities_eval() --m:load_weights_entities(true)
m:model_erect()
m:build_table()
     --m:train_entity_eval() --m:train_anchor_entity_eval()
m:train_entities_for_entity_TR('Barack_Obama')
     --m:store_train_entity_weight('eval')
]]----working

------------------TESTING-------------------------
--Load pre-trained entity vectors
--m:load_weights_entities()

		--Load pre-trained anchor based entity vectors
		--m:load_anchor_weights_entities(false)

	--test on get '5 similar words' task.
	--m:print_sim_words({"the","king","computer"},5)

--[[word1 = {'Michelle_Obama','United_States'} --'king'
word2 = {'Barack_Obama','Princeton_University', 'Harvard_Law_School'} --'queen'
--print(table.getn(word1))
for i=1,table.getn(word1) do --print(word1[i])
	for j=1,table.getn(word2) do
		sim = m:getSimilarity(word1[i], word2[j])
		if sim ~= nil then 
			print (string.format(' %s : %s = %0.6f', word1[i], word2[j], sim[1]))
		end
	end
end
]]--

--m:load_weights_entities(false)
--sim = m:getSimilarity('king', 'queen')
--if sim ~= nil then 
--	print (string.format(' cos similarity = %0.6f',  sim[1]))
--end

--m:testLinks('anchor1', 'anchor2')

--posList , negList = m:getPmiEntities('Japan', 4)
--pmi_value = m:getPMI('Barack_Obama', 'Michelle_Obama')
--print (string.format(' pmi = %s ', pmi_value))
--m:getPMIneighbour('Barack_Obama')



print (string.format(' This programs ran for %.2f seconds',sys.clock() - startTime ))
