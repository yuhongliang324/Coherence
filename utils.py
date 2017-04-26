__author__ = 'yuhongliang324'

import os, re
import numpy
from nltk.stem import WordNetLemmatizer

dn = os.path.dirname(os.path.abspath(__file__))

accident_data_root = os.path.join(dn, 'permutation/accident')
accident_train_root = os.path.join(accident_data_root, 'train')
accident_test_root = os.path.join(accident_data_root, 'test')

earthquake_data_root = os.path.join(dn, 'permutation/earthquake')
earthquake_train_root = os.path.join(earthquake_data_root, 'train')
earthquake_test_root = os.path.join(earthquake_data_root, 'test')

SUBJ, OBJ, OTHER = 3, 2, 1

from nltk.corpus import stopwords
# stop_words_list = set(stopwords.words('english'))
stop_words_list = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]


def load_grid(grid_file):
    reader = open(grid_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return numpy.zeros((2, 2))

    num_entity = len(lines)
    num_sent = len(lines[0].split()) - 1
    grid = numpy.zeros((num_sent, num_entity))

    for j in xrange(num_entity):
        sps = lines[j].split()
        sps = sps[1:]
        for i in xrange(num_sent):
            if sps[i] == '-':
                continue
            if sps[i] == 'S':
                grid[i, j] = SUBJ
            elif sps[i] == 'O':
                grid[i, j] = OBJ
            else:
                grid[i, j] = OTHER
    return grid


def load(root_path):
    files = os.listdir(root_path)
    files.sort()
    doc_grids = {}
    for fn in files:
        if not fn.endswith('.grid'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        if doc not in doc_grids:
            doc_grids[doc] = []
        fpath = os.path.join(root_path, fn)
        grid = load_grid(fpath)
        if grid is None:
            continue
        doc_grids[doc].append(grid)
    return doc_grids


def load_verb(root_path, merge=False):
    files = os.listdir(root_path)
    files.sort()
    doc_grids = {}
    verb_vocab = set()

    small_glove_file_path = os.path.join(root_path, 'glove.840B.300d_filtered.txt')
    if os.path.isfile(small_glove_file_path) and merge:
        small_glove = load_glove_model(small_glove_file_path)

    idx = 0
    for fn in files:
        if not fn.endswith('.parsed'):
            continue
        sp = fn.split('.')
        doc = '.'.join(sp[:2])
        if doc not in doc_grids:
            doc_grids[doc] = []
        fpath = os.path.join(root_path, fn)

        if os.path.isfile(small_glove_file_path) and merge:
            grid = load_grid_verb_merged(fpath, small_glove)
        else:
            grid = load_grid_verb(fpath, verb_vocab)

        if grid is None:
            continue
        doc_grids[doc].append(grid)

        print 'file: %d' % idx
        idx += 1

    if not os.path.isfile(small_glove_file_path):
        export_small_glove_model(root_path, verb_vocab)

    return doc_grids


pattern = re.compile(r'\(\w+? V[A-Z]*? "(.+?)"\)')
wnl = WordNetLemmatizer()


# Each paragraph has a grid
def load_grid_verb(grid_file, verb_vocab):
    reader = open(grid_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return numpy.zeros((2, 2))
    lines = lines[1:-1]

    num_sent = len(lines)
    verb_ID = {}
    curID = 0
    vIDs = []
    for i in xrange(num_sent):
        vID = set()
        verbs = re.findall(pattern, lines[i])
        verbs = map(lambda x: wnl.lemmatize(x.lower(), pos='v'), verbs)

        verb_vocab.update(verbs)

        for verb in verbs:
            if verb not in verb_ID:
                verb_ID[verb] = curID
                curID += 1
            vID.add(verb_ID[verb])
        vIDs.append(vID)

    num_entity = curID
    grid = numpy.zeros((num_sent, num_entity))

    for i in xrange(num_sent):
        vID = vIDs[i]
        for v in vID:
            grid[i][v] = 1.
    return grid


def load_grid_verb_merged(grid_file, small_glove):
    reader = open(grid_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    if len(lines) == 0:
        return numpy.zeros((2, 2))
    lines = lines[1:-1]

    num_sent = len(lines)
    verb_ID = {}
    ID_verb = {}
    curID = 0
    vIDs = []
    for i in xrange(num_sent):
        vID = set()
        verbs = re.findall(pattern, lines[i])
        verbs = map(lambda x: wnl.lemmatize(x.lower(), pos='v'), verbs)
        verbs = [v for v in verbs if '|' not in v] # For the earthquake dataset
        verbs = [v for v in verbs if v not in stop_words_list]


        for verb in verbs:
            if verb not in verb_ID:
                verb_ID[verb] = curID
                ID_verb[curID] = verb
                curID += 1
            vID.add(verb_ID[verb])
        vIDs.append(vID)

    vid2root, num_root = consider_sim_words(vIDs, ID_verb, small_glove)

    cluster_idx = 0
    root_set = set()
    root2cluster = {}
    for vid, root in vid2root.iteritems():
        if root not in root_set:
            root_set.add(root)
            root2cluster[root] = cluster_idx
            cluster_idx += 1

    grid = numpy.zeros((num_sent, num_root))

    for i in xrange(num_sent):
        vID = vIDs[i]
        for v in vID:
            grid[i][root2cluster[vid2root[v]]] += 1.
    return grid


def load_glove_model(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile, 'r')
    model = {}
    for idx, line in enumerate(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding

        if idx % 1000 == 0:
            print idx

    print "Done.", len(model), " words loaded!"
    return model


def export_small_glove_model(root_path, verb_vocab):
    for verb in verb_vocab:
        print verb

    glove_model = load_glove_model('../glove.840B.300d.txt')

    print 'length of verb_vocab: ', len(verb_vocab)
    # verb_vocab = list(verb_vocab)
    # verb_vocab.sort()

    with open(os.path.join(root_path, 'glove.840B.300d_filtered.txt'), 'w') as f:
        for idx, verb in enumerate(verb_vocab):
            print idx

            f.write(verb)

            embedding = glove_model.get(verb, ['NULL'])

            f.write(' ' + ' '.join([str(val) for val in embedding]) + '\n')


def consider_sim_words(vIDs, ID_verb, small_glove):
    oov_emb = calc_oov_emb(small_glove)

    graph_as_list = construct_graph(vIDs, ID_verb, small_glove, oov_emb)
    vid2root, num_root = count_components(graph_as_list)

    return vid2root, num_root


def calc_oov_emb(small_glove):
    sum_vec = numpy.zeros((1, 300))

    for k, v in small_glove.iteritems():
        sum_vec += v

    return sum_vec / len(small_glove)


def construct_graph(vIDs, ID_verb, small_glove, oov_emb):
    # Input: list of vIDs

    graph = {}
    # thres = 0.7 # Best for Cos Sim, acc=0.633
    thres = 4 # Best for Euc Dis, acc=0.663

    verb_list = []
    for id in xrange(len(ID_verb)):
        graph[id] = []
        verb_list.append(ID_verb[id])

    max_sim = 0
    min_sim = 1

    for idx1, v1 in enumerate(verb_list):
        for idx2 in xrange(idx1 + 1, len(verb_list)):
            v2 = verb_list[idx2]

            sim = calc_sim(small_glove.get(v1, oov_emb), small_glove.get(v2, oov_emb))
            if sim < thres:
                graph[idx1].append(idx2)
                graph[idx2].append(idx1)

                print 'word pair: %s, %s. Sim: %f' % (v1, v2, sim)

            max_sim = max(max_sim, sim)
            min_sim = min(min_sim, sim)

    graph_as_list = []
    for id in xrange(len(graph)):
        v = graph[id]
        graph_as_list.append([id, v])

    return graph_as_list


def calc_sim(emb1, emb2):
    emb1 = numpy.asarray(emb1).reshape(1, -1)
    emb2 = numpy.asarray(emb2).reshape(1, -1)

    # Cosine Sim
    # from sklearn.metrics.pairwise import cosine_similarity
    # return cosine_similarity(emb1, emb2)

    # Euclidean Distance
    # return numpy.exp(-numpy.linalg.norm(emb1 - emb2))
    return numpy.linalg.norm(emb1 - emb2)


def count_components(nodes):
    sets = {}
    for node in nodes:
        sets[node[0]] = DisjointSet(node[0])
    for node in nodes:
        for vtx in node[1]:
            sets[node[0]].union(sets[vtx])

    num_root = len(set(x.find() for x in sets.itervalues()))

    vid2root = {}
    for node in nodes:
        vid = node[0]
        vid2root[vid] = sets[vid].find().name

    return vid2root, num_root


class DisjointSet(object):
    def __init__(self, name):
        self.name = name
        self.parent = self

    def find(self):
        if self.parent is self:
            return self
        else:
            self.parent = self.parent.parent
            return self.parent.find()

    def union(self, other):
        them = other.find()
        us = self.find()
        if them != us:
            us.parent = them


def test_uf():
    nodes = [[1, [2, 3]], [2, [1]], [3, [4]], [4, [3]], [5, []], [6, []]]
    return count_components(nodes)
