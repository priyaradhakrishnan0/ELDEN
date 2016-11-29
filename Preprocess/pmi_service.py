from flask import Flask,jsonify,request,make_response,abort
import cPickle as pickle
import json
import math
from scipy import io
import time
from collections import OrderedDict
import heapq

#CO = pickle.load(open("", 'r'))

#stem_cache_file ="./data/pmi_data/stem_cache_1_2_16.pickle"
vocab_file = "vocab.pickle"
#prohibited_file = "./data/pmi_data/prohibited_1_2_16.pickle"
co_file = "/scratchd/home/priya/co-occuranceFiles/123_co.mtx" #"/scratch/home/priya/co-occuranceFiles/one_co.mtx" #new_co.mtx" # md5sum sent
#reverse_vocab_file = "./data/pmi_data/reverse_vocab.pickle"


vocab = pickle.load(open(vocab_file, 'r'))
#prohibited = pickle.load(open(prohibited_file, 'r'))
#stem_cache = pickle.load(open(stem_cache_file, 'r'))
#reverse_vocab = pickle.load(open(reverse_vocab_file, 'r'))

#To clip the number of list elements returned for the pmi list query
listLimit = 10

co = {}

app = Flask(__name__)


def convert(w):
    if w in prohibited or w in vocab:
        return w
    if len(w.split()) == 1:
        if w in stem_cache:
            return stem_cache[w]
        else:
            return None

    words = w.split()
    s = ""
    for i in words:
        if i not in stem_cache:
            return None
        s += stem_cache[i]
        s += " "
    return s[:-1]


def get_relevant_id(a):
    a = a.lower()
    c_a = convert(a)
    if c_a in vocab:
        return vocab[c_a]
    return -1


@app.route('/pmi/query', methods=['GET'])
def get_pmi():
    if("node" not in request.args):
        return jsonify({})

    input_string = request.args.get("node")
    print "pmi query for source node : ", input_string
    inp = input_string.split(";")

    if len(inp) != 2:
        abort(404)
    #a = get_relevant_id(inp[0])
    #b = get_relevant_id(inp[1])
    #if a != -1 and b != -1:
    if inp[0] in vocab and inp[1] in vocab:
        if co[vocab[inp[0]], vocab[inp[1]]] != 0 and co[vocab[inp[0]], vocab[inp[0]]] != 0 and co[vocab[inp[1]], vocab[inp[1]]] != 0:
            print 'finding for index ', vocab[inp[0]], vocab[inp[1]]
            answer = {}
            pmi_value = math.log(co[vocab[inp[0]], vocab[inp[1]]]  * len(vocab)/(1.0 * co[vocab[inp[0]], vocab[inp[0]]]*co[vocab[inp[1]], vocab[inp[1]]]))
            #answer["pmi"] = pmi_value
            #return jsonify(answer)
            return str(pmi_value)

    abort(404)
    return

#sends list of entities co-occuring with the given entity
@app.route('/pmi/list', methods=['GET'])
def get_pmi_list():
    if("node" not in request.args):
        return jsonify({})

    input_node = request.args.get("node")
    #print "pmi list query for source node : ", input_node
    ans_entities = {}
    if input_node in vocab:
        row = co.getrow(int(vocab[input_node])).nonzero()[1]
        if len(row) > 0:
            #ans_entities['length'] = len(row)
            for neighbour in row:
                #co[vocab[input_node], str(a)] is not 0
                #print 'neighbour =', neighbour
                if co[vocab[input_node], vocab[input_node]] != 0 and co[str(neighbour), str(neighbour)] != 0 :
                    pmi_value = math.log(co[vocab[input_node], str(neighbour)] * len(vocab)/(1.0 * co[vocab[input_node], vocab[input_node]] * co[str(neighbour), str(neighbour)]))
                    ans_entities[str(neighbour)] = pmi_value
                    #print neighbour, pmi_value
                    '''
                    #good to view results. But delays the query!!
                    for k,v in vocab.iteritems():
                        if v==str(neighbour):
                            #print k,pmi_value
                            ans_entities[k] = pmi_value
                            break;
                    '''
                    if listLimit > 0:
                        ans_entities_limited = {}
                        ans_entities_sort = heapq.nlargest(listLimit, ans_entities, key=ans_entities.get) #OrderedDict(sorted(ans_entities.items(), key=lambda t: t[1], reverse=True))
                        for topper in ans_entities_sort:
                            ans_entities_limited[topper] = ans_entities[topper] #set top listLimit entities 
                        ans_entities = ans_entities_limited #replace the original with limited count
            return jsonify(ans_entities)
        else:
            abort(404)#return None
    else:
        abort(404)#return None

    abort(404)
    return


@app.route('/neighbours/query', methods=['GET'])
def get_neighbours():
    if("node" not in request.args):
        return jsonify({})

    input_string = request.args.get("node")
    print "neighbour query for source node : ", input_string

    a = get_relevant_id(input_string)
    if a != -1:
        n = co.getrow(a).nonzero()[1]
        n_list = []
        for i in n:
            n_list.append(reverse_vocab[i][0])
        answer = {}
        answer["list"] = n_list
        return jsonify(answer)

    abort(404)
    return



app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'term not present'}), 404)

def main():
    global co
    start_time = time.time()
    co = io.mmread(co_file).tocsr()
    end_time = time.time()
    print 'PMI Server initialised in ', end_time - start_time, 'time'
    #app.run(host="0.0.0.0", port=int("2338"),debug = True,use_reloader=False,threaded=False)

    #app.run(host="0.0.0.0", port=int("2337"),debug = True,use_reloader=False,threaded=True)
    app.run(host="0.0.0.0", port=int("2340"),debug = True,use_reloader=False,threaded=True)


if __name__ == '__main__':
    main()
