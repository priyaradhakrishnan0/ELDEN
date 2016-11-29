dofile("word2vec.lua")

local app = require('waffle')
--PMI
--app.host = '10.16.32.104' --dosa --'10.16.32.103' --momo --app.host='10.16.34.115' --mall lab GPU   
--app.port = 1338
--WLM
app.host = '10.16.32.104' --dosa --'10.16.32.105' --vagrant --app.host = 
app.port = 1339 --7
--W0 Untrained
--app.host = '10.16.32.104' --dosa --103 momo
--app.port=1340 --1336


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

m = Word2Vec(config)

--load the trained weight files
--Trained weights
--PMI
--metadata_file = 'trainMetadataEntity.t7.eval5.2'--'trainMetadataEntity.t7.eval6' 
--weight_file = 'trainWeightEntity.W.eval5.2' --'trainWeightEntity.W.eval6' 
--WLM
metadata_file = 'trainMetadataEntity.t7.eval3' --'trainMetadataEntity.t7.eval5.2' --'trainMetadataEntity.t7.eval5.1'   --'trainMetadataEntity.t7.eval' --trainMetadataEntity.t7.0' --
weight_file = 'trainWeightEntity.W.eval3' --'trainWeightEntity.W.eval5.2' --'trainWeightEntity.W.eval5.1' --'trainWeightEntity.W.eval' --'trainWeightEntity.W.0'  --
--Raw weights (untrained)
--metadata_file = 'metadata_ED_wlm.t7'
--weight_file = 'finalWeight_ED_wlm.W'
m:load_weights_entities_EDServers(metadata_file, weight_file)

--[[app.get('/table/(%d+),(%d+),(%a+)', function(req, res)
   local row1 = req.params[1]
   local row2 = req.params[2]
   local row3 = req.params[3]
   print(string.format("row1 = %d ",row1))
   print(string.format("row2 = %d ",row2))
   print(string.format("row3 = %s ",row3))
   res.send(string.format('Pick row number %d, %d', row1,row2))
end)
]]--
--[[
app.get('/table/(%d+),(%d+)', function(req, res)
   local row1 = req.params[1]
   local row2 = req.params[2]
   print(string.format("row1 = %d ",row1))
   print(string.format("row2 = %d ",row2))
   res.send(string.format('Pick row number %d, %d', row1,row2))
end)
]]--


app.get('/ED/([%w-._,()]+):([%w-._,()]+)', function(req, res)
   local entity1 = req.params[1]
   local entity2 = req.params[2]
   --print(string.format("E1 = %s , E2 = %s ",entity1, entity2))

   sim = m:getSimilarity(entity1,entity2)   
	if sim ~= nil then 
		print (string.format(' %s : %s = %0.6f', entity1, entity2, sim[1]))
		--res.send (string.format(' %s : %s = %0.6f', entity1, entity2, sim[1]))
      res.send (string.format('%0.6f',sim[1]))
	else
		res.send (string.format('0.0'))
	end
	--res.send(string.format('Received E1 %s E2 %s', entity1, entity2))
end)


app.listen()
