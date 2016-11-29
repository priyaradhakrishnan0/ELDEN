--[[
Class for word2vec with skipgram and negative sampling
original source - https://github.com/yoonkim/word2vec_torch
modified - April 2016 by priya.r
--]]

--CPU version - July 20

require("sys")
require("nn")
--require("cunn")
--require("cutorch")
--require("cudnn")
--cutorch.setDevice(2)
--require("csvigo")
local http = require"socket.http"
local ltn12 = require"ltn12"
local json=require "cjson"

mongorover = require("mongorover")

--db parameters
local dbIP = '10.16.32.103' 
local dbPort = 27017
local client = mongorover.MongoClient.new("mongodb://"..dbIP..":"..dbPort)
LinkDb = client:getDatabase("wikilinks")
inlinks = LinkDb:getCollection("inlinks")
AnchorDb = client:getDatabase("anchorDB")
wikiTitleCategoryDb = client:getDatabase("wikiTitleCategoryDB")
anchors = AnchorDb:getCollection("anchors")
wikiPageDb = client:getDatabase("wikiPageDB")
wikiPageTitle = wikiPageDb:getCollection("wikiPageTitle")
wikiTitleCategory = wikiTitleCategoryDb:getCollection("wikiTitleCategory")
totalrecall = client:getDatabase("totalrecall")
pmi = totalrecall:getCollection("pmi")

local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
    torch.setdefaulttensortype('torch.FloatTensor')
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
    self.dim = config.dim
    self.criterion = nn.BCECriterion() -- logistic loss
    self.word = torch.IntTensor(1) 
    self.contexts = torch.IntTensor(1+self.neg_samples) 
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
    self.window = config.window 
    self.lr = config.lr 
    self.min_lr = config.min_lr
    self.alpha = config.alpha
    self.table_size = config.table_size 
    --self.vocab = {}
    self.sample_freq = 0 --empirically set
    self.index2word = {}
    self.word2index = {}
    self.total_count = 0
    self.total_count_pow = 0
    self.wordTokens = 1246881 --emperically set. from create_gensim_weights_entities()
    --print('Initialized wor2vec instance')
    self.mlp = nn.CosineDistance()
    --print ('constructor called!!', self.criterion)
end

-- move to cuda
function Word2Vec:cuda()
    if(config.gpu>0) then
        cutorch.setDevice(config.gpu)
    end
    --cutorch.setDevice(2) -- for MALL lab GPU 2 (Tesla)
    print(string.format("Using gpu %d",cutorch.getDevice()))
    cutorch.setHeapTracking(true)
    self.word = self.word:cuda()
    self.contexts = self.contexts:cuda()
    self.labels = self.labels:cuda()
    self.criterion:cuda()
    self.w2v:cuda()
    collectgarbage()
end

--Get wikipedia pagele (i.e. entity) of this pageId or nil
function Word2Vec:getPageTitle(qId)
    local pg = nil
    docs = wikiPageTitle:find_one({pageId=qId}, {title=true, _id=false})
    if docs ~= nil then 
        for k,v in pairs(docs) do 
            --print(k,v)
            pg = v
        end
    end
    return (pg)
end

--Get wikipedia pageId of this title or 0
function Word2Vec:getPageId(qTitle)
    local  pgId = 0
    --qTitle="Michelle_Obama"
    --print(qTitle)     
    --docs = wikiPageTitle:count({title=qTitle})
    --print(docs)
    docs = wikiPageTitle:find_one({title=qTitle}, {pageId=true, _id=false})
    if docs ~= nil then 
        for k,v in pairs(docs) do 
            --print(k,v)
            pgId = v
        end
    end    
    --docs = wikiPageTitle:find({title=qTitle}, {pageId=true, _id=false})
    --[[for d in docs do 
        print (d) 
        print (d.pageId)
        pgId = d.pageId--pgId = v 
    end
    --]]
    return (pgId)
end

function Word2Vec:testLinks(anchor1, anchor2)
    print(string.format('%s has %d inlinks ',anchor1, getInlinkFreq(anchor1)))
    local posList1, negList1 = {} , {}--print(string.format('%s has inlinks ',anchor1)) 
    local anchorList1 = getInlinks(anchor1)
    for k,v in pairs(anchorList1) do
        posList1 = getPagesWithLink(k)
    end
    local anchorList2 = getInlinks(anchor2)
    print (string.format('anch list 2 = %d',getTableLength(anchorList2))) 
    for k,v in pairs(anchorList2) do
        posList2 = getPagesWithLink(k)
        for k1,v1 in pairs(posList2) do
            if type(posList1[k1]) == nil then 
                negList1[k1] = 0
                print (string.format('Hurray : Neg Node = %s',k1)) 
            end
        end
    end
    print (string.format('Pos Samples = %d',getTableLength(posList1))) 
    --[[
    print("Titles")
    print(tablelength(getTitles('World_Digital_Library_related')))
    print("categories")
    entity = 'Burkina_Faso'
    print(tablelength(getCategories(entity)))
    for i,cat in ipairs(getCategories(entity)) do
        print(cat)
    end
    ]]
end

--Get wikipedia categories of this title
function getCategories(qtitle)
    local docs = wikiTitleCategory:find_one({title=qtitle}, {categories=true,_id=false})
    if docs ~= nil then
        for k,v in pairs(docs) do 
            --print(k,v)
            return v
        end
    else 
        return 0
    end
end

--Get wikipedia titles under this category
function getTitles(Category)
    local titles = {}
    local docs = wikiTitleCategory:find({categories=Category}, {title=true,_id=false})
    if docs ~= nil then
        for doc in docs do
            titles[doc.title] = 0
        end
    end
    return titles
end

function getInlinkFreq (anchor)
    local inlinkFreq = 0
    local linksReturned = inlinks:find({page=anchor},{link=true, _id=false})
    for res in linksReturned do 
        for k,v in pairs(res) do
            for i,link in pairs(v) do
                inlinkFreq = inlinkFreq + 1
                --print(i,link)
            end
        end
    end
    return inlinkFreq
end

function getInlinks (anchor)
    local inlinkList = {}
    --local inlinkCount = 1
    local linksReturned = inlinks:find({page=anchor},{link=true, _id=false})
    for res in linksReturned do 
        for k,v in pairs(res) do
            for i,link in pairs(v) do
                inlinkList[link] = 0 --inlinkList[inlinkCount] = link
                --inlinkCount = inlinkCount + 1
            end
        end
    end
    return inlinkList
end

function Word2Vec:getPmiEntities(qEntity, limit)
    --Query pmi table for this entity. All entities with pmi > 0.0 are pos samples and with pmi < 0.0 are neg samples
    local posList = {}
    local negList = {}
    local  posCount = 0
    local negCount = 0 
    local recs = pmi:find({entity1=qEntity},{entity2=true, pmi=true, _id=false})
    rec_count = 0
    for rec in recs do
        printFlag = false
        print(string.format(" rec # %d",rec_count))
        for k,v in pairs(rec) do
            --print(k,v)
            if k == 'pmi' then
                pmi_value = tonumber(v)
                --if pmi_value > 0 then
                --    printFlag = true
                --end
            end  
            if k == 'entity2' then
            	if pmi_value > 0 then
            		posList[v] = pmi_value
            		posCount = posCount + 1
            	else
            		negList[v] = pmi_value
            		negCount = negCount + 1
            	end
                --print (string.format("e = %s pmi = %s",v,pmi_value))                
            end 
        end
        rec_count = rec_count + 1
        if posCount > limit and negCount > 1 then break end
    end
    print("POS list")
    printTable(posList)
    print("NEG list")
	printTable(negList)
    return posList, negList
end

--load word2index prior to calling this 
--return table of indexes
function Word2Vec:getCommonLinkedEntities(entity, limit) --One hop entity neighbours
    --1.find inlinks to entiy
    local inlinkList = getInlinks(entity)
    --print(string.format('%s has %d inlinks', entity, tablelength(inlinkList)))
    --2.find pages for which these are inlinks
    local pageList = {}
    local pageCount = 0
    for link, i in pairs(inlinkList) do
        --print (string.format('Inlink : %s',link))
        posList = getPagesWithLink(link)
        if tablelength(posList) > 0 then
            for page_name, i in pairs(posList) do
                page_idx = self.word2index[page_name]
                if page_idx ~= nil then 
                    --print(string.format('pos page : %s',page_name))
                    pageCount = pageCount +1
                    pageList[page_idx] = 0--pageList[page_name] = 0
                    if pageCount >= limit then break end
                end
            end
        end
        if pageCount >= limit then break end        
    end

    --print(string.format('POS list length = %d', tablelength(pageList)))
    return pageList
end

function getPagesWithLink (link)
    local pageList = {}
    --local inlinkCount = 1
    local linksReturned = inlinks:find({link=link},{page=true, _id=false})
    for res in linksReturned do 
        pageList[res.page] = 0 --inlinkList[inlinkCount] = link
                --inlinkCount = inlinkCount + 1
    end
    --print (string.format('pages with inlink to %s = %d', link, tablelength(pageList)))
    return pageList
end

function printTable(myTable)
    for k,v in pairs(myTable) do
        print (k,v)
    end
end

function getTableLength(myTable)
    local rowCount =0
    for k,v in pairs(myTable) do
        rowCount = rowCount + 1
    end
    return rowCount
end

function getCommonLinks (anchor1, anchor2)
    local commonLinks = {}
    local inlinkList1 = getInlinks(anchor1)
    local inlinkList2 = getInlinks(anchor2)
    for k,v in pairs(inlinkList1) do
        if type(inlinkList2[v]) ~= nil then
            commonLinks[v] = 0
        end
    end
    return commonLinks
end

-- Lua implementation of PHP scandir function
function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls -a "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = "./"..directory.."/"..filename
    end
    pfile:close()
    return t
end

--find number of lines in a file
function getLineCount(FName)
    local lineCount, popen = 0, io.popen
    local pfile = popen('wc -l "'..FName..'"')
    for line in pfile:lines() do
        --print(line)
        if line ~= '\n' then 
            lineCount = line:split(' ')[1] + 0 --adding 0 makes the str to int
        end
    end
    --print (lineCount)
    return lineCount
end

--number of all entries. table.getn(table_name) and #table_name will give only count entries with integer keys,
function tablelength(Tablename)
  local count = 0
  for _ in pairs(Tablename) do count = count + 1 end
  return count
end

--find anchors-texts that point to this page-id
function Word2Vec:getAnchorList(qPgid)
    anchList = {}
    --print(string.format('Anchors of %d',qPgid))
    for result in anchors:find({["pages.page_id"] = qPgid}) do
        --print(string.format("%s ",result.anchor))
        anchList[result.anchor]=0
    end

    --[[
    for retAnchor in anchors:find() do
        for k,v in pairs(retAnchor.pages) do
            for k1,v1 in pairs(v) do   --print("k1,v1 = ",k1,v1)
                if k1=="page_id" then
                    if v1==qPgid then
                        --print(string.format("%s ",retAnchor.anchor))
                        anchList[retAnchor.anchor]=0
                    end
                end
            end
        end
    end
    ]]--
    --print(string.format('#anchors = %d',tablelength(anchList)))
    return anchList
end

-- Build vocab frequency, word2index, and index2word from input file
function Word2Vec:build_vocab(vocab)
    print("Building vocabulary indices from vocab file...")
    local start = sys.clock()
    --local n = 1
    local f = io.open(vocab, "r")
    local t={} 
    if f then
        for line in f:lines() do
            t = {}; i = 1
            for s in string.gmatch(line, "%S+") do
                t[i] = s
                i = i + 1
            end
            if table.getn(t) == 2 then
                --print(t)
                self.total_count = self.total_count + 1
                -- Delete words that do not meet the minfreq threshold and create word indices
                self.sample_freq = tonumber(t[2])
                if self.sample_freq >= self.minfreq then
                    self.index2word[#self.index2word+1] = t[1]
                    self.word2index[t[1]] = #self.index2word 
                    self.total_count_pow = self.total_count_pow + self.sample_freq^self.alpha
        	    end
            end
            --n = n + 1
            --if self.total_count % 1000000 == 0 then
                --print(" Lines : %d",self.total_count,", after cleaning : ",#self.index2word)
            --end
        end
        f:close()
        print (" vocab lines = ", self.total_count)
    end        
    print ("total count %d",self.total_count_pow," average word prob = ",self.total_count_pow/self.total_count) 
    self.vocab_size = #self.index2word               
    print(string.format("%d words mappings processed in %.2f seconds.", self.total_count, sys.clock() - start))
    print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))    
    -- initialize word/context embeddings now that vocab size is known
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
    self.context_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings
    self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
    --self:model_erect()
end

--Erect the model
function Word2Vec:model_erect()
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
    self.w2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)

    --print ('model erected', self.w2v)
end

--Load pre-trained entity embeddings
function Word2Vec:load_weights_entities_cpu(trainFlag)
    --local numEntitys = 4661047 --4900000 --wikiPageTitle:count();    
    --self.vocab_size = getLineCount(config.pretrain) + numEntitys
    self.vocab_size = 6822839
    print (string.format('W * d = %d * %d',self.vocab_size, self.dim)) -- 6822839    100
    self.index2word, self.word2index = unpack(torch.load('trainMetadataEntity.t7'))
    print (' metadata loaded for entity')
    
    --for training create the model and copy weight tensor into it. For testing just load the weight tensor.
    if trainFlag then --for training create the model and copy weight tensor into it. For testing just load the weight tensor.
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        self.word_vecs.weight:copy(torch.load('trainWeightEntity_cpu.W'))
        --self.word_vecs = torch.load('trainWeightEntity_cpu.W')
        print ('Word weights loaded for entity ',self.word_vecs:size())
        self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        self.context_vecs.weight:copy(torch.load('trainWeightEntity_cpu.H'))
        --self.context_vecs = torch.load('trainWeightEntity_cpu.H')
        print ('Context weights loaded for entity ',self.context_vecs:size())
        collectgarbage()
    else--testing. Loads only W
        --self.word_vecs = torch.load('trainWeightEntity_cpu.W') 
        self.word_vecs = torch.load('trainWeightEntity_cpu.W')
        collectgarbage()
        print ('Word weights loaded for entity testing ',self.word_vecs:size())
    end
end

--Copying 250,000 rows at a time as There is a lue limit in the number of constants in a single function at 2^18 i.e 262144. http://lua-users.org/lists/lua-l/2008-02/msg00298.html
function Word2Vec:copy_chunks(start_position,end_position)
    --print (string.format('Copying %d to %d ', start_position,end_position))
    self.word_vecs.weight[{{start_position,end_position}}]:copy(self.Wcopy[{{start_position,end_position}}])
end

--Load pre-trained entity embeddings
function Word2Vec:load_weights_entities_eval()
    self.vocab_size = 5910721 --eval3 = numTokens + numEntitys = 4663840 + 1246881

    metadata_file = 'trainMetadataEntity.t7.eval3'--'trainMetadataEntity.t7.0'
    weight_file = 'trainWeightEntity.W.eval3'--'trainWeightEntity.W.0'
    hidden_file = 'trainWeightEntity.H.eval3'--'trainWeightEntity.H.0'

    print (string.format('W * d = %d * %d',self.vocab_size-1, self.dim)) -- 6822839    100
    self.index2word, self.word2index = unpack(torch.load(metadata_file))
    print (' metadata loaded for entity')

    self.word_vecs = nn.LookupTable(self.vocab_size-1, self.dim) --:cuda()
    self.word_vecs.weight:copy(torch.load(weight_file))
    --self.word_vecs = torch.load('trainWeightEntity.W')
    --self.word_vecs = torch.zeros(6822838,100)
    print ('Word weights loaded for entity ',self.word_vecs.weight:size())
    collectgarbage()

    self.context_vecs = nn.LookupTable(self.vocab_size-1 , self.dim) --:cuda()
    self.context_vecs.weight:copy(torch.load(hidden_file))
    --self.context_vecs = torch.load('trainWeightEntity.H')
    --self.context_vecs = torch.zeros(6822838,100)
    print ('Context weights loaded for entity ',self.context_vecs.weight:size())
    collectgarbage()
end

--Load pre-trained entity embeddings
function Word2Vec:load_weights_entities(trainFlag)
    --local numEntitys = 4661047 --4900000 --wikiPageTitle:count();    
    --self.vocab_size = getLineCount(config.pretrain) + numEntitys

    --for Titles2.csv, vocab_size = 5678734 from finalWeight_small.W and metadata_small.t7
    --[[self.vocab_size = 5678734 --1712985 --6822839 vocab_size = 1712985 (1651698 + wikiEntityWordCount = 61286)
    metadata_file = 'trainMetadataEntity.t7.0'
    weight_file = 'trainWeightEntity.W.0'
    hidden_file = 'trainWeightEntity.H.0'
    ]]-- -

    --for Titles3.csv, vocab_size = 5910721 from finalWeight_ED_wlm.W and metadata_ED_wlm.t7
    self.vocab_size = 5910721
    metadata_file = 'trainMetadataEntity.t7.eval3'
    weight_file = 'trainWeightEntity.W.eval3'
    hidden_file = 'trainWeightEntity.H.eval3'

    print (string.format('W * d = %d * %d',self.vocab_size, self.dim)) -- 6822839    100
    self.index2word, self.word2index = unpack(torch.load(metadata_file))
    print (' metadata loaded for entity')

    if trainFlag then 
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim) --:cuda()
        if self.vocab_size < 262144 then 
            self.word_vecs.weight:copy(torch.load(weight_file))
        else 
            runs = math.ceil(self.vocab_size/262144) --no of runs
            chunk = math.ceil(self.vocab_size/runs) --chunk_size
            self.Wcopy = torch.load(weight_file)
            i = 0
            while i < (runs - 1) do
                self:copy_chunks((i*chunk)+1, (i+1)*chunk) 
                i = i + 1
            end
            self:copy_chunks(((runs-1)*chunk)+1, self.vocab_size-1)
            self.Wcopy = nil
        end
        --self.word_vecs = torch.load('trainWeightEntity.W')
        --self.word_vecs = torch.zeros(6822838,100)
        print ('Word weights loaded for entity ',self.word_vecs.weight:size())
        collectgarbage()

        self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim) --:cuda()
        self.context_vecs.weight:copy(torch.load(hidden_file))
        --self.context_vecs = torch.load('trainWeightEntity.H')
        --self.context_vecs = torch.zeros(6822838,100)
        print ('Context weights loaded for entity ',self.context_vecs.weight:size())
        collectgarbage()
    else--testing. Loads only W
        self.word_vecs = torch.load(weight_file)
        collectgarbage()
        print ('Word weights loaded for entity testing ',self.word_vecs:size())
        local type = torch.typename(self.word_vecs)
        print (string.format('Type of word vec is %s ',type))
        if type == 'torch.FloatTensor' or type == 'torch.DoubleTorch' then
            print (string.format('Its not Cuda. Its %s ',type))
        end

        ---START CPU loadable
        --[[now = sys.clock()
         a=torch.load('trainWeightEntity.W'):float()
         torch.save('trainWeightEntity_cpu.W', a)
        type = torch.typename( a)
        
        print (string.format('New type of word vec is %s ',type))
        if type == 'torch.FloatTensor' or type == 'torch.DoubleTensor' then
            print (string.format('Its not Cuda. Its %s ',type))
        end
        print(string.format(' Stored the trained weights in %.2f seconds ', sys.clock() - now))
        ]]--
        --END CPU loadable

        --self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        --self.context_vecs.weight:copy(torch.load('trainWeight.H'))
        --collectgarbage()
        --print ('context weights loaded')
    end

end


--Load pre-trained entity embeddings from cpu training
function Word2Vec:load_weights_entities_EDServers(metadata_file, weight_file)

    self.index2word, self.word2index = unpack(torch.load(metadata_file))    
    --print (string.format('W * d = %d * %d',#self.index2word, self.dim)) -- 6822839    100
    self.word_vecs = torch.load(weight_file)
    print (string.format(' metadata %s and weight %s loaded for EDserver', metadata_file, weight_file))
    collectgarbage()
    --print ('Word weights loaded for entity testing ',self.word_vecs:size())
    --local type = torch.typename(self.word_vecs)
    --print (string.format('Type of word vec is %s ',type))

    --if type == 'torch.FloatTensor' or type == 'torch.DoubleTorch' then
    --    print (string.format('Its not Cuda. Its %s ',type))
    --end
end

---load anchor trained weights prior to entity training
--trainFlag=true is training
function Word2Vec:load_anchor_weights_entities(trainFlag)
    --local numEntitys = 4661047 --4900000 --wikiPageTitle:count();    
    --self.vocab_size = getLineCount(config.pretrain) + numEntitys
    self.vocab_size = 6822839

    print (string.format('W * d = %d * %d',self.vocab_size, self.dim)) -- 6822839    100

    if trainFlag then 
        self.index2word, self.word2index = unpack(torch.load('trainMetadata.t7'))
        print ('anchor metadata loaded')
        
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        self.word_vecs.weight:copy(torch.load('trainWeight.W'))
        collectgarbage()
        print ('anchor word weights loaded')

        self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        self.context_vecs.weight:copy(torch.load('trainWeight.H'))
        collectgarbage()
        print ('anchor context weights loaded')

    else --testing. Loads only W
        self.index2word, self.word2index = unpack(torch.load('trainMetadata.t7'))
        print ('anchor metadata loaded for testing')
        
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        self.word_vecs.weight:copy(torch.load('trainWeight.W'))
        collectgarbage()
        print ('anchor word weights loaded for testing')

        --self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        --self.context_vecs.weight:copy(torch.load('trainWeight.H'))
        --collectgarbage()
        --print ('context weights loaded')
    end
end

--trainFlag=true is training
function Word2Vec:load_gensim_weights_entities(trainFlag)
    --local numEntitys = 4661047 --4900000 --wikiPageTitle:count();    
    --self.vocab_size = getLineCount(config.pretrain) + numEntitys  
    self.vocab_size = 5910721 --5678734 --1712985 --5678734--6822839 --for 1/10 Entities, vocab_size = 1712985 (1651698 + wikiEntityWordCount = 61286); vocab_size = 5910720 (4663840 entities + 1246880 word_embeddings );
    print (string.format('W * d = %d * %d',self.vocab_size, self.dim)) -- 6822839    100

    if trainFlag then 
        self.index2word, self.word2index = unpack(torch.load('metadata_ED_wlm.t7')) --metadata_small.t7')) --metadata_small_1.t7')) --metadata.t7'))
        print ('Gensim metadata loaded')
        
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim) --:cuda()
        self.word_vecs.weight:copy(torch.load('finalWeight_ED_wlm.W')) --finalWeight_small.W'))--finalWeight_small_1.W')) --finalWeight.W'))
        collectgarbage()
        print ('Gensim word weights loaded')

        self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim) --:cuda()
        self.context_vecs.weight:copy(torch.load('finalWeight_ED_wlm.H')) --finalWeight_small.H')) --finalWeight_small_1.H')) --finalWeight.H'))
        collectgarbage()
        print (' Gensim context weights loaded')

    else --testing. Loads only W
        self.index2word, self.word2index = unpack(torch.load('metadata_ED_wlm.t7')) --metadata_small.t7')) --metadata.t7'))
        print ('Gensim  metadata loaded for testing')
        
        self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim) --:cuda()
        self.word_vecs.weight:copy(torch.load('finalWeight_ED_wlm.W')) --finalWeight_small.W')) --finalWeight.W'))
        --self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim)
        --self.word_vecs.weight:copy(torch.load('trainWeight_cpu.W'))
        collectgarbage()
        print ('Gensim word weights loaded for testing')

        ---START CPU loadable
        --[[now = sys.clock()
        torch.save('trainWeight_cpu.W', self.word_vecs:float().weight)
        print(string.format(' Stored the trained weights in %.2f seconds ', sys.clock() - now))
        ]]--
        --END CPU loadable

        --self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim):cuda()
        --self.context_vecs.weight:copy(torch.load('trainWeight.H'))
        --collectgarbage()
        --print ('context weights loaded')
    end
end

--Load gensim wordvectors as pre-trained vectors and instantiate embeddings for wiki entities
--The weight matrix created thus is stored. So call this function only once.
function Word2Vec:create_gensim_weights_entities()

    local start = sys.clock()
    local numEntitys = 4663840 --math.floor(4661047/10) --4661047 --4900000 --wikiPageTitle:count(); <<+change here for ALL_ENTITIES to 4661047, ALL_ENTITIES from Titles3 = 4663840.
    local numTokens = 1246881 --1017687 --1246881 -- Emperically determined with goodWords as '^[%w_%-,().]+$' and not instantiating for wikiEntityWordCount

    torch.setdefaulttensortype('torch.FloatTensor') --to create _small weight files. Using halfTensors of 16 bits instead of float of 32 precision and double's 64 precision

    print(string.format(" No of Entities = %d, No of Tokens = %d", numEntitys, numTokens))
    self.vocab_size = numTokens + numEntitys --self.vocab_size = getLineCount(config.pretrain) + numEntitys
     print(string.format(" word + entities = %d", self.vocab_size)); --self.vocab_size = 1000
    -- initialize word/context embeddings now that vocab size is known
    self.word_vecs = nn.LookupTable(self.vocab_size -1, self.dim) -- word embeddings
    print(string.format(' W lookup table instantiation took %.2f secs.',sys.clock() - start))
    local now = sys.clock()
    self.context_vecs = nn.LookupTable(self.vocab_size -1, self.dim) -- context embeddings
    self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
    print(string.format(' H Lookup table initialization took %.2f seconds.', sys.clock() - now))
    --fGensim = csvigo.load{path='model_small.csv', mode = "large"}
    --fGensim = csvigo.load{path=config.pretrain, mode = "large"}
    print(string.format('Starting index2word %d word2index %d', #self.index2word, #self.word2index))
    local fGen = torch.DiskFile(config.pretrain, "r")
    fGen:quiet()
    local line = fGen:readString('*l') -- read header
    line = fGen:readString('*l') -- read first word
    --print(line)
    local lineCount = 2 --2 lines read so far
    local totalWordCount = getLineCount(config.pretrain)
    local quoteCount = 0
    local tokenCount = 1
    while ( lineCount <= totalWordCount ) do
        if string.match(line,'".*"') ~=nil then
            --wordStr = string.match(line, '".*"')
            --print (string.format("Correcting needed for %s", wordStr))
            quoteCount = quoteCount + 1
            --line = string.gsub(line, wordStr,''); --line = string.gsub(line, '".*"','')
            --values = line:split(',')
            --values[1] = wordStr
        else 
            values = line:split(',')
            --print (string.format("Word is %s",values[1]))
            if values[1] ~= nil then --this if condition fires only in model_small.csv                
                --check for alphanumeric strings alone
                if string.match(values[1],'^[%w_%-,().]+$') ~= nil then 
                    featureTensor = torch.Tensor(100)
                    for j=2, #values do
                        --print(string.format("i=%d, j=%d, val=%.2f",lineCount,j,values[j]))
                        featureTensor[j-1] = values[j]                
                    end
                    --print(string.format("i=%d, val=%s, #f=%d",lineCount,values[1], #values-1))
                    self.word_vecs.weight[tokenCount]:copy(featureTensor)
                    self.index2word[tokenCount] = values[1]
                    self.word2index[values[1]] = tokenCount
                    --if lineCount%10000==0 then print(values[1],lineCount) end --for status
                    tokenCount = tokenCount + 1
                end                
            end
        end
        lineCount = lineCount + 1
        line = fGen:readString('*l')
        if line == nil then break end ---if lineCount == 100 then break end
    end        
    
    fGen:close()
    print(string.format(' Loaded %d word embeddings from gensim word2vec in %.2f seconds. Ignored %d due to quotes. Tokens = %d ', #self.index2word, sys.clock() - start, quoteCount, tokenCount))
     
    now = sys.clock()   
    local fTitle = torch.DiskFile('Titles3.csv', 'r') --Titles2.csv', 'r')
    fTitle:quiet()
    --local entity = fTitle:readString('*l') print(entity)
    wikiEntityWordCount = 0
    local entity_count =0
    while entity_count < numEntitys do
        entity = fTitle:readString('*l') --print(entity)
        entity_count = entity_count + 1
        entity_idx = self.word2index[entity]
        if entity_idx == nil then
            position = #self.index2word + 1
            featureTensor = torch.Tensor(100)
            if position >= self.vocab_size then 
                print(string.format('Breaking at index %s and entity %d out of %d ', position, entity_count, numEntitys))
                break
            end
            self.word_vecs.weight[position]:copy(featureTensor)
            self.index2word[position] = entity
            self.word2index[entity] = position
            --entity_count = entity_count + 1
            --print(position)
        else
            --print(string.format('Found wikiWord for entity %s',entity))
            wikiEntityWordCount = wikiEntityWordCount +1
        end
    end
    fTitle:close()
    print(string.format(' Instantiated word embeddings for %d entities in %.2f seconds. wikiEntityWordCount = %d ', entity_count, sys.clock() - now, wikiEntityWordCount))
    now = sys.clock()
    torch.save('finalWeight_ED_wlm.W', self.word_vecs.weight)
    torch.save('finalWeight_ED_wlm.H', self.context_vecs.weight)
    torch.save('metadata_ED_wlm.t7', {self.index2word, self.word2index})
    print(string.format(' Saved word and hidden embeddings of length %d in %.2f seconds ', #self.index2word, sys.clock() - now))
end

--Lua table length
function Word2Vec:tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- Loading stored lookup weights from input weight file
function Word2Vec:load_weights()
    print("Loading trained weights...")
    w2 = torch.load(config.embeddings)
    print('Weights in saved model : ')
    --print(w2)
    print(w2:size()[1])
    self.vocab_size = w2:size()[1]
    -- initialize word/context embeddings now that vocab size is known
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- word embeddings
    self.context_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings
    self.word_vecs:reset(0.25); self.context_vecs:reset(0.25) -- rescale N(0,1)
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
    self.w2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
end

-- Store word embeddings
function Word2Vec:store_wordEmbeddings()
    print("Saving trained weights to weightFile")
    --print(self.word_vecs.weight)
    torch.save(config.embeddings, self.word_vecs.weight)
end

-- Build a table of unigram frequencies from which to obtain negative samples
function Word2Vec:build_table()
    local start = sys.clock()
    print("Building a table of unigram frequencies... ")
    self.table = torch.IntTensor(self.table_size)
    local word_index = 1
    --local random_word_prob = self.sample_freq^self.alpha / self.total_count_pow --original
    --optionally this can be average_word_prob
    local random_word_prob = 0.00038 --.1 --emperical found average_word_prob as 3.8
    local word_prob = random_word_prob  
    for idx = 1, self.table_size do
        self.table[idx] = word_index
        if idx / self.table_size > word_prob then
           word_index = word_index + 1
	       word_prob = word_prob + random_word_prob
        end
        if word_index > self.vocab_size then
           word_index = word_index - 1
        end
        --print(idx..','..word_index)
    end
    --print(string.format("Done in %.4f seconds.",  sys.clock() - start))
end

-- Train on word context pairs
function Word2Vec:train_pair(word, contexts)
    local p = self.w2v:forward({contexts, word})
    local loss = self.criterion:forward(p, self.labels)
    local dl_dp = self.criterion:backward(p, self.labels)
    self.w2v:zeroGradParameters()
    self.w2v:backward({contexts, word}, dl_dp)
    self.w2v:updateParameters(self.lr)
end


-- Batch train on word context pairs
function Word2Vec:train_batch(word_batch, contexts_batch, labels_batch)
    --print (word_batch:size())
    --print (contexts_batch:size())
    --print (labels_batch:size())
    --print (self.w2v)
    --print (self.criterion)
    local p = self.w2v:forward({contexts_batch, word_batch})
    local loss = self.criterion:forward(p, labels_batch)
    local dl_dp = self.criterion:backward(p, labels_batch)
    self.w2v:zeroGradParameters()
    self.w2v:backward({contexts_batch, word_batch}, dl_dp)
    self.w2v:updateParameters(self.lr)
end

-- Sample negative contexts
function Word2Vec:sample_contexts(context)
    self.contexts[1] = context
    local i = 0
    while i < self.neg_samples do
        neg_context = self.table[torch.random(self.table_size)]
    	if context ~= neg_context then
    	    self.contexts[i+2] = neg_context
    	    i = i + 1
    	end
    end
end

-- Sample negative entities
function Word2Vec:sample_entities(pos_idx)
    self.contexts[1] = pos_idx
    local i = 0
    while i < self.neg_samples do
        neg_entity_idx = torch.random(self.wordTokens,self.vocab_size)
        if (pos_idx ~= neg_entity_idx and self.index2word[neg_entity_idx] ~= nil) then
            self.contexts[i+2] = neg_entity_idx
            i = i + 1
        end
    end
end

-- Sample negative pmi entities
function Word2Vec:sample_pmi_entities(pos_idx,pmiTab)
    self.contexts[1] = pos_idx
    local i = 0
    while i < self.neg_samples do
        neg_entity_idx = torch.random(self.wordTokens,self.vocab_size)        --
        if (pos_idx ~= neg_entity_idx and self.index2word[neg_entity_idx] ~= nil) then
            if (pmiTab[neg_entity_idx] ~= nil) then
                if (pmiTab[neg_entity_idx] < 0) then 
                    self.contexts[i+2] = neg_entity_idx
                    i = i + 1
                end
            else
                self.contexts[i+2] = neg_entity_idx
                i = i + 1
            end
        end
    end
end

-- Train on neighbour nodes in the kb
function Word2Vec:train_entity()
    if self.gpu>0 then
        self:cuda()
    end
    print("Training entity...")
    local start = sys.clock()

    local batch_size = 100 --1000s 
    local word_batch = torch.FloatTensor(batch_size, 1):zero() --local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local c = 1
    local count = 1

    for epoch=1,config.epochs do
        local fGen = torch.DiskFile(config.entityTrain, "r")
        fGen:quiet()
        local entity
        local posSamples = {}
        local negSamples = {}
        local totalWordCount = getLineCount(config.entityTrain)     print(string.format("Collected %d entities for training", math.floor(1.0*totalWordCount/3)))
        local missingWcount , entity_idx = 0, 0

        local line = fGen:readString('*l')
        local lineCount, skipC  =  1 , 0
        while ( lineCount <= totalWordCount ) do

            if string.match(line,' , ') == nil and lineCount%3 == 1 then
                entity = line  
                entity_idx = self.word2index[entity] --print(string.format("Processing %s ( %d )", entity, entity_idx))
            else 
                values = line:split(' , ')
                if lineCount%3 == 2 then --POS samples
                    skipC = 0
                    for j=1, #values do  --print(string.format("i=%d, j=%d, val=%s",lineCount,j,values[j]))
                        if self.word2index[values[j]] == nil then
                            skipC = skipC + 1
                        else 
                           posSamples[j-skipC] = self.word2index[values[j]]
                        end
                    end
                elseif lineCount%3 == 0 then --NEG samples
                    skipC = 0
                    for j=1, #values do  --print(string.format("i=%d, j=%d, val=%s",lineCount,j,values[j]))
                        if self.word2index[ values[j]] == nil then
                            skipC = skipC + 1
                        else
                            negSamples[j - skipC] = self.word2index[values[j]]
                        end
                    end
                end
                --print(string.format("After line %d with %d pos and %d neg samples.", lineCount, #posSamples, #negSamples))
            end

            if lineCount %3 ==0 then 
                --print(string.format("Processing %s with %d pos and %d neg samples.", entity, #posSamples, #negSamples))
                if entity_idx ~= nil and #posSamples > self.window and #negSamples > (self.neg_samples * self.window) then 
                    print(string.format("Processing %s with %d pos and %d neg samples.", entity, #posSamples, #negSamples))
                    self.word[1] = entity_idx 
                    for i=1,self.window do
                        self.contexts[1] = posSamples[i]
                        for j=1, self.neg_samples do
                            self.contexts[j+1] = negSamples[(i-1)*5 + j]
                        end

                        contexts_batch[c]:copy(self.contexts)
                        word_batch[c]:copy(self.word)
                        labels_batch[c]:copy(self.labels)
                        c = c + 1
                        self.lr = math.max(self.min_lr, self.lr + self.decay) 
                        if c  > batch_size then                                        
                            self:train_batch(word_batch, contexts_batch,labels_batch)
                            count = count + c
                            print(string.format("%d words from %.2f entities trained in %.2f seconds. Learning rate: %.4f. Missed W count = %d", count, lineCount/3, sys.clock() - start, self.lr, missingWcount))
                            --reset
                            c = 1
                            --word_batch = torch.CudaTensor(batch_size, 1):zero()
                            --contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
                            --labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
                            collectgarbage()
                        end
                    end  
                elseif entity_idx == nil then
                    missingWcount = missingWcount + 1
                end
                --reset to re-use
                entity_idx = 0
                posSamples = {}
                negSamples = {} 
            end

            if lineCount%1000==0 then 
                print(entity,lineCount)
                collectgarbage() 
            end
            lineCount = lineCount + 1
            line = fGen:readString('*l')
            if line == nil then break end

        end --end-while

        print(string.format(" EPOCH %d: %d entities with %d contexts trained in %.2f seconds. Missing W entries for %d", epoch, math.floor(1.0*totalWordCount/3), count, sys.clock() - start, missingWcount ))
        if c > 1 then --process the remaining words in batch
            print(string.format("remaining words = %d",c))
            self:train_batch(word_batch, contexts_batch,labels_batch)
            c = 1
            --word_batch = torch.CudaTensor(batch_size, 1):zero()
            --contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
            --labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
        end 
        fGen = nil
        count = 1
        collectgarbage()
        print(string.format('Missing  W entry for %d entities in epoch %d ',missingWcount, epoch))
    end --end of epoch

    print ('Freeing space now..')
    --freeing some mem
    word_batch = nil
    contexts_batch = nil
    labels_batch = nil
    --self.w2v:clearState() --REVISIT
    collectgarbage()
    --self:store_train_entity_weight()
    print ('Done with train_entity()')
end

-- Train on common category in the kb
function Word2Vec:train_entity_cat() --<<<START HERE
    if self.gpu > 0 then
        self:cuda()
    end
    print("Training entity, by shared category links..")
    local start = sys.clock()

    local batch_size = 100 --1000s 
    local word_batch = torch.FloatTensor(batch_size, 1):zero() --local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local c = 1
    local count = 1

    for epoch=1,config.epochs do
        local fGen = torch.DiskFile(config.entityTrain, "r")
        fGen:quiet()
        local entity
        local posSamples = {}
        local negSamples = {}
        local totalWordCount = getLineCount(config.entityTrain)     print(string.format("Collected %d entities for training", math.floor(1.0*totalWordCount/3)))
        local missingWcount , entity_idx = 0, 0

        local line = fGen:readString('*l')
        local lineCount, skipC  =  1 , 0
        while ( lineCount <= totalWordCount ) do

            if string.match(line,' , ') == nil and lineCount%3 == 1 then
                local entity = line  
                local entity_idx = self.word2index[entity] --print(string.format("Processing %s ( %d )", entity, entity_idx))
                --populate posSamples from shared category
                if getCategories(entity) ~= 0 then
                    for i,cat in ipairs(getCategories(entity)) do
                        if tablelength(getTitles(cat)) > 0 then
                            ---ha ha ...got the pos titles
                        end
                    end
                else
                    print(string.format('Title %s cannot be processed for EDcat, 0 categories',entity))
                end

                --populate neg entity from Word2Vec:sample_entity(pos_entity_idx)
            
                --print(string.format("After line %d with %d pos and %d neg samples.", lineCount, #posSamples, #negSamples))
            end

            if lineCount %3 ==0 then 
                --print(string.format("Processing %s with %d pos and %d neg samples.", entity, #posSamples, #negSamples))
                if entity_idx ~= nil and #posSamples > self.window and #negSamples > (self.neg_samples * self.window) then 
                    print(string.format("Processing %s with %d pos and %d neg samples.", entity, #posSamples, #negSamples))
                    self.word[1] = entity_idx 
                    for i=1,self.window do
                        self.contexts[1] = posSamples[i]
                        for j=1, self.neg_samples do
                            self.contexts[j+1] = negSamples[(i-1)*5 + j]
                        end

                        contexts_batch[c]:copy(self.contexts)
                        word_batch[c]:copy(self.word)
                        labels_batch[c]:copy(self.labels)
                        c = c + 1
                        self.lr = math.max(self.min_lr, self.lr + self.decay) 
                        if c  > batch_size then                                        
                            self:train_batch(word_batch, contexts_batch,labels_batch)
                            count = count + c
                            print(string.format("%d words from %.2f entities trained in %.2f seconds. Learning rate: %.4f. Missed W count = %d", count, lineCount/3, sys.clock() - start, self.lr, missingWcount))
                            --reset
                            c = 1
                            --word_batch = torch.CudaTensor(batch_size, 1):zero()
                            --contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
                            --labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
                            collectgarbage()
                        end
                    end  
                elseif entity_idx == nil then
                    missingWcount = missingWcount + 1
                end
                --reset to re-use
                entity_idx = 0
                posSamples = {}
                negSamples = {} 
            end

            if lineCount%1000==0 then 
                print(entity,lineCount)
                collectgarbage() 
            end
            lineCount = lineCount + 1
            line = fGen:readString('*l')
            if line == nil then break end

        end --end-while

        print(string.format(" EPOCH %d: %d entities with %d contexts trained in %.2f seconds. Missing W entries for %d", epoch, math.floor(1.0*totalWordCount/3), count, sys.clock() - start, missingWcount ))
        if c > 1 then --process the remaining words in batch
            print(string.format("remaining words = %d",c))
            self:train_batch(word_batch, contexts_batch,labels_batch)
            c = 1
            --word_batch = torch.CudaTensor(batch_size, 1):zero()
            --contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
            --labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
        end 
        fGen = nil
        count = 1
        collectgarbage()
        print(string.format('Missing  W entry for %d entities in epoch %d ',missingWcount, epoch))
    end --end of epoch

    print ('Freeing space now..')
    --freeing some mem
    word_batch = nil
    contexts_batch = nil
    labels_batch = nil
    --self.w2v:clearState() --REVISIT
    collectgarbage()
    --self:store_train_entity_weight()
    print ('Done with train_entity()')
end

function Word2Vec:store_train_entity_weight()
    now = sys.clock()
    torch.save('trainWeightEntity.W', self.word_vecs.weight)
    torch.save('trainWeightEntity.H', self.context_vecs.weight)
    torch.save('trainMetadataEntity.t7', {self.index2word, self.word2index})
    print(string.format(' Stored the trained weights in %.2f seconds ', sys.clock() - now))    
end

--stores the weight vecs as FloatTensor to be merged in CPU.
function Word2Vec:store_train_entity_weight(sliceNum)
    now = sys.clock()
    torch.save('trainWeightEntity.W.'..sliceNum, self.word_vecs.weight:float())
    torch.save('trainWeightEntity.H.'..sliceNum, self.context_vecs.weight:float())
    torch.save('trainMetadataEntity.t7.'..sliceNum, {self.index2word, self.word2index})
    --print(string.format(' Stored the trained weights in %.2f seconds ', sys.clock() - now))    
end

--Trains anchors for entities in config.entityTrain
function Word2Vec:train_anchor_entity()
    print("Training anchors for entities...")
    --.1. collect entities
    local start = sys.clock()
    local  trainingEntities = {}
    local fGen = torch.DiskFile(config.entityTrain, "r")
    fGen:quiet()
    local totalWordCount = getLineCount(config.entityTrain)     print(string.format("Collected %d entities for training", math.floor(1.0*totalWordCount/3)))
    local line = fGen:readString('*l')
    local lineCount =  1
    while ( lineCount <= totalWordCount ) do
        if string.match(line,' , ') == nil and lineCount%3 == 1 then
            trainingEntities[line] = 0
        end
        lineCount = lineCount + 1
        line = fGen:readString('*l')
        if line == nil then break end
    end --end-while
    fGen = nil
    collectgarbage()
    --.2. train anchors for the collected entities
    for entity_name, ent_count in pairs(trainingEntities) do
        self:train_anchor_for_entity(entity_name)
    end
end

--TAC and AIDA dataset entities in EDqueries.csv and evalEntitiesMap.txt
function Word2Vec:train_entity_eval()
    print("Training anchors for entities in TAC and AIDA...")
    --.1. collect entities dealt with in evaluation
    local  trainingEntities={}
    --reading from EDqueries.csv
    local fGen = torch.DiskFile('EDqueries.csv', "r") --EDqueries.csv_bkup', "r")
    fGen:quiet()
    local totalWordCount = getLineCount('EDqueries.csv')     
    local line = fGen:readString('*l')
    local lineCount = 1
    while ( lineCount <= totalWordCount ) do
        if string.match(line,':::') ~= nil then
            local list = self:split(line,':::')
            for i,entitypair in ipairs(list) do
                --print(entitypair)
                for str in string.gmatch(entitypair, "([^#]+)") do
                    --print(str)
                    trainingEntities[str] = 0;
                end
            end
        end
        lineCount = lineCount + 1
        line = fGen:readString('*l')
        --if lineCount == 100 then break end
    end --end-while
    fGen = nil
    --end reading from EDqueries.csv
    --reading from evalEntitiesMap.txt 
    fGen = torch.DiskFile('evalEntitiesMap.txt', "r")
    fGen:quiet()
    local numEvalEntitys = getLineCount('evalEntitiesMap.txt')
    local entity_count = 0
    while entity_count < numEvalEntitys do
        entity = fGen:readString('*l') --print(entity)
        entity_count = entity_count + 1
        if trainingEntities[entity] == nil then
            trainingEntities[entity] = 0
        end
    end
    fGen = nil
    collectgarbage()    
    print(string.format("Read %d entities of evaluation.", tablelength(trainingEntities)))
    --.2. train anchors for the collected entities
    local entityCount, pmi_trained_entity_count = 0, 0
    for epoch=1,config.epochs do
        local start = sys.clock()
        for entity_name, ent_count in pairs(trainingEntities) do
            
            --[[EDwlm start
            self:train_anchor_for_entity(entity_name)
            self:train_entities_for_entity(entity_name)
            --EDwlm end ]]
            
            --EDpmi start
            if self:train_entities_for_entity_TR(entity_name) == true then
                pmi_trained_entity_count = pmi_trained_entity_count + 1
            end
            --EDpmi end

            entityCount = entityCount + 1
            if entityCount%100 == 0 then 
                self:store_train_entity_weight('eval6')
                print(string.format('STATUS : Trained %d out of %d entities for eval-epoch %d, %d had pos-pmi-entries, took %0.2f seconds', entityCount, tablelength(trainingEntities), epoch, pmi_trained_entity_count, sys.clock() - start)) 
            end
        end 
        self:store_train_entity_weight('eval6')
        print(string.format('EPOCH : %d, Runtime : %0.2f seconds', epoch, sys.clock() - start))
    end
end

--Enhance Titles2 to Titles3 with entities missing in W
function Word2Vec:addMissingTitles()
    print("Checking for entities in TAC and AIDA, missing in W")
    --0. Take stock of existing entities
    local existingTitles = {} 
    local fTitle = torch.DiskFile('Titles2.csv', 'r')
    fTitle:quiet()
    --local entity = fTitle:readString('*l') print(entity)
    local numEntitys = getLineCount('Titles2.csv')
    local entity_count =0
    while entity_count < numEntitys do
        entity = fTitle:readString('*l') --print(entity)
        entity_count = entity_count + 1
        existingTitles[entity] = 0
    end
    --.1. collect entities dealt with in evaluation
    local  trainingEntities={}
    --reading from EDqueries.csv
    local fGen = torch.DiskFile('EDqueries.csv_bkup', "r")
    fGen:quiet()
    local totalWordCount = getLineCount('EDqueries.csv')     
    local line = fGen:readString('*l')
    local lineCount = 1
    while ( lineCount <= totalWordCount ) do
        if string.match(line,':::') ~= nil then
            local list = self:split(line,':::')
            for i,entitypair in ipairs(list) do
                --print(entitypair)
                for str in string.gmatch(entitypair, "([^#]+)") do
                    --print(str)
                    trainingEntities[str] = 0;
                end
            end
        end
        lineCount = lineCount + 1
        line = fGen:readString('*l')
        --if lineCount == 100 then break end
    end --end-while
    fGen = nil
    --end reading from EDqueries.csv
    --reading from evalEntitiesMap.txt 
    fGen = torch.DiskFile('evalEntitiesMap.txt', "r")
    fGen:quiet()
    local numEvalEntitys = getLineCount('evalEntitiesMap.txt')
    local entity_count = 0
    while entity_count < numEvalEntitys do
        entity = fGen:readString('*l') --print(entity)
        entity_count = entity_count + 1
        if trainingEntities[entity] == nil then
            trainingEntities[entity] = 0
        end
    end
    fGen = nil
    collectgarbage()    
    print(string.format("Read %d entities of evaluation.", tablelength(trainingEntities)))
    --.2. Add the training entities if not present in Titles2
    fWrite = io.open('Titles3.csv', "wb" )
    local existCount = 0
    local  newCount = 0
    for k,v in pairs(existingTitles) do
        fWrite:write(k.."\n")
        existCount = existCount + 1
    end    
    for k,v in pairs(trainingEntities) do
        if existingTitles[k] == nil then
            fWrite:write(k.."\n")
            newCount = newCount + 1
        end
    end
    fWrite:close()
    print(string.format("Added %d old and %d new entities in Titles3.csv", existCount, newCount))
end -- Added 4661045 old and 2773 new entities in Titles3.csv


-- Train on anchor-texts of an entity from the anchors collection of anchorDb
function Word2Vec:train_anchor_for_entity(entity)
    --if self.gpu>0 then
    --    self:cuda()
    --end
    --print(string.format("Training anchors of %s", entity))
    --list = scandir(corpus)
    self.train_words = {}; self.train_contexts = {}
    local start = sys.clock()

    local batch_size = 25--1000 
    local word_batch = torch.FloatTensor(batch_size, 1):zero()--local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    
    local c = 1 
    local count = 0
    local anchList = {}
    local qPgId = self:getPageId(entity)
    if qPgId ~= 0 then
        --print(string.format("Entiy pg %d",qPgId))
        local entity_idx = self.word2index[entity]
        --print(string.format("Entiy idx %d",entity_idx))

        if entity_idx ~= nil then 
            self.word[1] = entity_idx

            local anchList = self:getAnchorList(qPgId)     --2204744);
            if (tablelength(anchList) > 0) then 
                --print(string.format("Collected %d anchors", tablelength(anchList)))
                missingWcount = 0
                for anchor_text,ac in pairs(anchList) do   
                    --print ("Current anchor : ",anchor_text)                    
                    local contexts = self:split(anchor_text)
                    for i, context in ipairs(contexts) do
                        local context_idx = self.word2index[context] --print("Current context word : ",context)
                        if context_idx ~= nil then -- valid context
                            --print (c,entity_idx,context_idx)
                            self:sample_contexts(context_idx) -- update pos/neg contexts
                            contexts_batch[c]:copy(self.contexts)
                            word_batch[c]:copy(self.word)
                            labels_batch[c]:copy(self.labels)
                            c = c + 1
                            self.lr = math.max(self.min_lr, self.lr + self.decay) 
                            if c  > batch_size then                                        
                                self:train_batch(word_batch, contexts_batch,labels_batch)
                                count = count + c
                                --print(string.format("%s : %d words from %d anchors trained in %.2f seconds. Learning rate: %.4f. Missed W count = %d", entity, count, i, sys.clock() - start, self.lr, missingWcount))
                                c = 1
                                missingWcount = 0
                                --collectgarbage()
                            end
                        else
                            missingWcount  = missingWcount + 1
                        end
                    end
                end
                
                c = c - 1 --to account for the last addition to c( index) alone
                if c > 1 then --process the remaining words in batch
                    --print(string.format("remaining words = %d",c))

                    for i=1,c do
                        --print(string.format("Processing %d",i))
                        self:train_pair(word_batch[i], contexts_batch[i])
                    end
                    --c = 1
                end 
                --print(string.format(" %s : %d anchors trained in %.2f seconds", entity, count+c, sys.clock() - start ))
            end
        end
    end -- no pgId in wiki 
end --end of train_anchor_per_entity()

--Train an entity in terms of its pos and neg entities, by WLM
function Word2Vec:train_entities_for_entity(entity)
    --print(string.format("Training POS and NEG of %s", entity))
    self.train_words = {}; self.train_contexts = {}
    local start = sys.clock()

    local batch_size = 5
    local word_batch = torch.FloatTensor(batch_size, 1):zero()--local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    
    local c = 1 
    local posList = {}
    local qPgId = self:getPageId(entity)
    if qPgId ~= 0 then
        --print(string.format("Entiy pg %d",qPgId))
        local entity_idx = self.word2index[entity]
        --print(string.format("Entiy idx %d",entity_idx))

        if entity_idx ~= nil then 
            self.word[1] = entity_idx

            local posList = self:getCommonLinkedEntities(entity, self.window) --One hop entity neighbour indexes
            if (tablelength(posList) > 0) then 
                --print(string.format("Collected %d anchors", tablelength(anchList)))
                for pos_idx, ac in pairs(posList) do   
                    --print (c,entity_idx,context_idx)
                    self:sample_entities(pos_idx) -- update pos/neg entities
                    contexts_batch[c]:copy(self.contexts)
                    word_batch[c]:copy(self.word)
                    labels_batch[c]:copy(self.labels)
                    c = c + 1
                    --self.lr = math.max(self.min_lr, self.lr + self.decay) 
                    if c  > batch_size then                                        
                        self:train_batch(word_batch, contexts_batch,labels_batch)
                        --print(string.format("%d pos entities trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
                        c = 1
                        collectgarbage()
                    end                  
                end
                
                c = c - 1 --to account for the last addition to c( index) alone
                if c > 1 then --process the remaining words in batch                    
                    --print(string.format("remaining entities = %d",c))
                    for i=1,c do
                        --print(string.format("Processing %d",i))
                        self:train_pair(word_batch[i], contexts_batch[i])
                    end                    
                end 
                --print(string.format("%s : %d pos entities trained in %.2f seconds", entity, c, sys.clock() - start ))
            end
        end
    end -- no pgId in wiki 
end --end of train_entities_for_entity(entity)

--Train an entity in terms of its pos and neg entities, by PMI
function Word2Vec:train_entities_for_entity_TR(entity)
    --print(string.format("Training POS and NEG of %s by PMI", entity))
    self.train_words = {}; self.train_contexts = {}
    local returnFlag = false
    local start = sys.clock()

    local batch_size = 60 --empirically fixed . refer  Statstics_eval_traing.txt
    local word_batch = torch.FloatTensor(batch_size, 1):zero()--local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.FloatTensor(batch_size, 1+self.neg_samples):zero() --local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    
    local c, count = 1 , 0
    local pmiTab, negList = {}, {}
    local qPgId = self:getPageId(entity)
    if qPgId ~= 0 then
        --print(string.format("Entiy pg %d",qPgId))
        local entity_idx = self.word2index[entity]
        --print(string.format("Entiy idx %d",entity_idx))

        if entity_idx ~= nil then 
            self.word[1] = entity_idx

            local pmiTab = self:getPMIneighbour(entity) --One hop entity neighbour index with PMI
            if pmiTab ~= nil then 
                --print(string.format("Collected %d neighbours", tablelength(pmiTab)))
                for idx, pmi_value in pairs(pmiTab) do   
                    --print (c,entity_idx,context_idx)
                    if tonumber(pmi_value) > 0 then
                        self:sample_pmi_entities(idx,pmiTab) -- update pos/neg entities
                        contexts_batch[c]:copy(self.contexts)
                        word_batch[c]:copy(self.word)
                        labels_batch[c]:copy(self.labels)
                        c = c + 1
                        --self.lr = math.max(self.min_lr, self.lr + self.decay) 
                        if c  > batch_size then                                        
                            self:train_batch(word_batch, contexts_batch,labels_batch)
                            --print(string.format("%d pos entities trained in %.2f seconds. Learning rate: %.4f", c-1, sys.clock() - start, self.lr))
                            count = count + c
                            c = 1
                            collectgarbage()
                        end                  
                    end
                end
                
                c = c - 1 --to account for the last addition to c( index) alone
                if c > 1 then --process the remaining words in batch                    
                    --print(string.format("remaining entities = %d",c))
                    for i=1,c do
                        --print(string.format("Processing %d",i))
                        self:train_pair(word_batch[i], contexts_batch[i])
                    end                    
                end 
                --print(string.format("%s : %d pos entities trained in %.2f seconds", entity, count+c, sys.clock() - start ))
                returnFlag = true
            else
            	--print(string.format("%s : No POS entities found",entity))
            end
        end
    end -- no pgId in wiki
    return returnFlag 
end --end of train_entities_for_entity(entity) by pmi

-- Train on anchor-texts and their linking entities from the anchors collection of anchorDb
function Word2Vec:train_anchor()
    if self.gpu>0 then
        self:cuda()
    end
    print("Training anchors...")
    --list = scandir(corpus)
    self.train_words = {}; self.train_contexts = {}
    local count = 0
    local start = sys.clock()

    local batch_size = 1000 
    local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    
    local c = 1 
    local anchCount = 0
    local anchList = {}
    local missingWcount = 0
    local missingW = {}
    local numAnchors = anchors:count();     print(string.format(" No of anchors = %d", numAnchors))

    for anchor in anchors:find() do
        anchCount = anchCount + 1
        anchList[anchCount] = anchor.anchor
        if anchCount>=500000 then break end --emperically fixed at 500,000. Increasing this gives memory error at the save time.
    end
    print(string.format("Collected %d anchors", #anchList))

    for epoch=1,config.epochs do
        missingWcount = 0
        for ac, anchor_text in pairs(anchList) do
            for retAnchor in anchors:find({anchor=anchor_text}) do        
                for k,v in pairs(retAnchor.pages) do
                      --print("k,v = ",k,v)
                    for k1,v1 in pairs(v) do   --print("k1,v1 = ",k1,v1)
                        if k1=="page_id" then --print("c = ",c)
                            local entity = self:getPageTitle(v1) --print(entity,anchor_text)
                            local entity_idx = self.word2index[entity]

                            if entity_idx ~= nil then 
                                self.word[1] = entity_idx
                                local contexts = self:split(anchor_text)
                                for i, context in ipairs(contexts) do
                                    local context_idx = self.word2index[context] --print("Current context word : ",context)
                                    if context_idx ~= nil then -- valid context
                                        --print (c,entity_idx,context_idx)
                                        self:sample_contexts(context_idx) -- update pos/neg contexts
                                        contexts_batch[c]:copy(self.contexts)
                                        word_batch[c]:copy(self.word)
                                        labels_batch[c]:copy(self.labels)
                                        c = c + 1
                                        self.lr = math.max(self.min_lr, self.lr + self.decay) 
                                        if c  > batch_size then                                        
                                            self:train_batch(word_batch, contexts_batch,labels_batch)
                                            count = count + c
                                            print(string.format("%d words from %d anchors trained in %.2f seconds. Learning rate: %.4f. Missed W count = %d", count, ac, sys.clock() - start, self.lr, missingWcount))
                                            c = 1
                                            --collectgarbage()
                                        end
                                    end
                                end
                            else
                                missingWcount = missingWcount + 1
                                --missingW[missingWcount] = entity --print(string.format('No W entry for %s ',entity))
                            end
                        end
                    end            
                end
            end 
            if ac%100 == 0 then  collectgarbage() end               
        end

        print(string.format(" EPOCH %d: %d anchors with %d contexts trained in %.2f seconds", epoch, anchCount, count, sys.clock() - start ))
        if c > 1 then --process the remaining words in batch
            print(string.format("remaining words = %d",c))
           self:train_batch(word_batch, contexts_batch,labels_batch)
           c = 1
        end 

        print(string.format('NO W ENTRY FOR %d',missingWcount))
        --print(missingW)

    end --end of epoch

    --freeing some mem
    anchList = {}
    word_batch = nil
    contexts_batch = nil
    labels_batch = nil
    self.w2v:clearState()
    collectgarbage()

    now = sys.clock()
    torch.save('trainWeight.W', self.word_vecs.weight)
    torch.save('trainWeight.H', self.context_vecs.weight)
    torch.save('trainMetadata.t7', {self.index2word, self.word2index})
    print(string.format(' Stored the trained weights in %.2f seconds ', sys.clock() - now))
    
end --end of train_anchor()

-- Train on sentences that are streamed from the hard drive
-- Check train_mem function to train from memory (after pre-loading data into tensor)
function Word2Vec:train_stream(corpus)
    print("Training...")
    list = scandir(corpus)
    self.train_words = {}; self.train_contexts = {}

    local batch_size = 1000 --400000 was emperically fixed for wikiData training
    local word_batch = torch.CudaTensor(batch_size, 1):zero()
    local contexts_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local labels_batch = torch.CudaTensor(batch_size, 1+self.neg_samples):zero()
    local c = 1
        
    for k,fileName in pairs(list) do
        if string.match(fileName, "xx") then
            local count = 0
            print("Processing contexts from ",fileName)
            local f = io.open(fileName, "r")
            local start = sys.clock()
            for line in f:lines() do
                sentence = self:split(line)
                for i, word in ipairs(sentence) do
        	       word_idx = self.word2index[word]
        	       if word_idx ~= nil then -- word exists in vocab
            	        local reduced_window = torch.random(self.window) -- pick random window size
        		        self.word[1] = word_idx -- update current word
                        for j = i - reduced_window, i + reduced_window do -- loop through contexts
        	                local context = sentence[j]
        		            if context ~= nil and j ~= i then -- possible context
        		                context_idx = self.word2index[context]
                                --print("Current context word : ",context)
        			            if context_idx ~= nil then -- valid context
                                    --print (c..','..context_idx)
          		                    self:sample_contexts(context_idx) -- update pos/neg contexts
                                    contexts_batch[c]:copy(self.contexts)
                                    word_batch[c]:copy(self.word)
                                    labels_batch[c]:copy(self.labels)
        			                 c = c + 1
        			                 self.lr = math.max(self.min_lr, self.lr + self.decay) 
        			                 if c  > batch_size then                                        
                                        self:train_batch(word_batch, contexts_batch,labels_batch)
                                        count = count + c
        			                     print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", count, sys.clock() - start, self.lr))
                                         c = 1
        			                 end
        			             end
        		            end
                        end		
        	        end
                end
            end
            print(string.format(" File %s with %d contexts trained in %.2f seconds", fileName, count, sys.clock() - start ))
        end
    end
    
    if c > 1 then --process the remaining words in batch
        print(string.format("remaining words = %d",c))
       self:train_batch(word_batch, contexts_batch,labels_batch)
    end 
end

--Get vector embedding similarity (cosine), true ON training, false for testing
function Word2Vec:getSimilarity(word1, word2, trainFlag)
    local w1
    local w2
    if self.word2index[word1] == nil then
       print(string.format('%s does not exist in vocabulary.',word1))
       return nil
    elseif self.word2index[word2] == nil then
       print(string.format('%s does not exist in vocabulary.',word2))
       return nil
    else
        --print(string.format('%s has index %d',word1, self.word2index[word1]))
        --print(string.format('%s has index %d',word2, self.word2index[word2]))
        if trainFlag then --In training, word_vec model is setup. Its weight contain weights
            w1 = self.word_vecs.weight[self.word2index[word1]]
            w2 = self.word_vecs.weight[self.word2index[word2]]
        else --In testing, word_vecs is just weight tensor
            w1 = self.word_vecs[self.word2index[word1]]
            w2 = self.word_vecs[self.word2index[word2]]
        end
        --print (string.format('dim of %s\'s in embeddings is %d ',word1,w1:nDimension()))
        --print (string.format('size of %s\'s in embeddings is %d',word1, w1:size()[1])) --print (w1:size())
        --print (string.format('size of %s\'s in embeddings is %d',word2, w2:size()[1])) --print (w2:size())
        ---for DOSA
        local sim1 = self.mlp:forward({w1, w2})
        --print (sim1)
        --For MALL GPU
        --local sim = self.mlp:forward({nn.utils.recursiveType(w1, 'torch.DoubleTensor'), nn.utils.recursiveType(w2, 'torch.DoubleTensor')})
        --print (sim)
        return sim1
    end
end

-- Row-normalize a matrix
function Word2Vec:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Word2Vec:get_sim_words(w, k)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    if type(w) == "string" then
        if self.word2index[w] == nil then
	       print("'"..w.."' does not exist in vocabulary.")
	       return nil
    	else
                w = self.word_vecs_norm[self.word2index[w]]
    	end
    end
    local sim = torch.mv(self.word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {self.index2word[idx[i]], -sim[i]}
    end
    return r
end

-- print similar words
function Word2Vec:print_sim_words(words, k)
    for i = 1, #words do
    	r = self:get_sim_words(words[i], k)
    	if r ~= nil then
       	    print("-------"..words[i].."-------")
    	    for j = 1, k do
    	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
    	    end
    	end
    end
end

-- split on separator
function Word2Vec:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    --print(input)
    --print(string.format('seperator is %s',sep))
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        --print(str)
        t[i] = str; i = i + 1
    end
    return t
end

-- pre-load data as a torch tensor instead of streaming it. this requires a lot of memory, 
-- As the corpus is huge, it is partition into smaller sets and the directory name is passed here
function Word2Vec:preload_data(corpus)
    print("Preloading training corpus into tensors (Warning: this takes a lot of memory)")
    local start = sys.clock()
    local c = 0
    list = scandir(corpus)
    self.train_words = {}; self.train_contexts = {}
    for k,fileName in pairs(list) do
        if string.match(fileName, "xx") then
            print("Processing contexts from ",fileName)
            local f = io.open(fileName, "r")
            for line in f:lines() do
            --f = io.open(corpus, "r")
            --for line in f:lines() do
                sentence = self:split(line)
                for i, word in ipairs(sentence) do
            	    word_idx = self.word2index[word]
            	    if word_idx ~= nil then -- word exists in vocab
                    	local reduced_window = torch.random(self.window) -- pick random window size
                		self.word[1] = word_idx -- update current word
                        for j = i - reduced_window, i + reduced_window do -- loop through contexts
                	       local context = sentence[j]
                		    if context ~= nil and j ~= i then -- possible context
                		        context_idx = self.word2index[context]
                    			if context_idx ~= nil then -- valid context
                    			    c = c + 1
                      		        self:sample_contexts(context_idx) -- update pos/neg contexts
                    			    if self.gpu>0 then
                    			        self.train_words[c] = self.word:clone():cuda()
                    			        self.train_contexts[c] = self.contexts:clone():cuda()
                    			    else
                    				    self.train_words[c] = self.word:clone()
                    				    self.train_contexts[c] = self.contexts:clone()
                    			    end
                    			end
                		    end
                        end	      
            	    end
            	end
            end
        end
        print(string.format("%d word-contexts processed in %.2f seconds", c, sys.clock() - start))
    end
end

-- train from memory. this is needed to speed up GPU training
function Word2Vec:train_mem()
    local start = sys.clock()
    for i = 1, #self.train_words do
        self:train_pair(self.train_words[i], self.train_contexts[i])
	self.lr = math.max(self.min_lr, self.lr + self.decay)
	if i%100000==0 then
            print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", i, sys.clock() - start, self.lr))
	end
    end    
end

-- train the model using config parameters
function Word2Vec:train_model(corpus)
    if self.gpu>0 then
        self:cuda()
    end
    if self.stream==1 then
        self:train_stream(corpus)
    else
        self:preload_data(corpus)
	self:train_mem()
    end
end

--Acessing pmi_sevice web API
function Word2Vec:getPMI(entity1, entity2)
    print(string.format("Query for %s, %s", entity1, entity2))
    urlString = 'http://10.16.32.104:2337/pmi/query?node='..entity1..';'..entity2
    local respbody = {} -- for the response body
    local result, respcode = http.request{url=urlString, sink=ltn12.sink.table(respbody)}
    if tonumber(respcode) == 200 then
        --print (string.format(" pmi-value = %s ",respbody[1]))
        return respbody[1]
    else
        return nil
    end
end

--neighbour nodes based on pmi_sevice web API
function Word2Vec:getPMIneighbour(entity)
    --print(string.format("PMI neighbours of %s", entity))
    urlString = 'http://10.16.32.104:2337/pmi/list?node='..entity
    local respbody = {} -- for the response body
    local result, respcode = http.request{url=urlString, sink=ltn12.sink.table(respbody)}
    if tonumber(respcode) == 200 then
        chunkNum = 1
        nextChunk = true
        response_text = ''
        while nextChunk == true do
            response_text = response_text..respbody[chunkNum]
            if string.match(respbody[chunkNum], '}') then
                nextChunk = false
            else
                chunkNum = chunkNum + 1
            end
        end

        tab = json.decode(response_text)
        if tablelength(tab) > 0 then
            --print(string.format('Found %s pmi-entity neighbours',tablelength(tab)))

            --for entity_idx, pmi_value in pairs(tab) do 
            --    print (string.format(" %s %s", entity_idx, pmi_value))
            --end
  
            return tab
        end
    else
        return nil
    end
end