from pycorenlp import StanfordCoreNLP

def unitoStr(word):
	if( type(word) != str):
		x = uni.normalize('NFKD', word).encode('ascii','ignore')
	else:
		x = word
	return x

def find_deptree(plaintext):
    nlp = StanfordCoreNLP("http://127.0.0.1:9000")

    output = nlp.annotate(plaintext, properties={'annotators': 'depparse', 'outputFormat': 'json'})
    parse = []
    group = output['sentences'][0]
    for i in range(len(output['sentences'][0]['basicDependencies'])):
    	dep = output['sentences'][0]['basicDependencies'][i]['dep']
    	dependent = output['sentences'][0]['basicDependencies'][i]['dependent']
    	govgloss = output['sentences'][0]['basicDependencies'][i]['governorGloss']
    	governor = output['sentences'][0]['basicDependencies'][i]['governor']
    	depengloss = output['sentences'][0]['basicDependencies'][i]['dependentGloss']
    	n = dependent
    	word = depengloss
    	tag = output['sentences'][0]['tokens'][int(n)-1]['pos']
    	head = governor
    	rel = dep
    	parse.append((int(n),word,tag,int(head),rel))
    sort = sorted(parse)
    visited = [0]*len(sort)
    dep_tree = []
    for i in range(len(sort)):
    	if sort[i][4] == 'ROOT':
    		root_index = i+1
    		visited[i] = 1
    		dep_tree.append( (("ROOT","ROOT",-1.1),"ROOT",(sort[i][1],sort[i][2],i)) )
    for i in range(len(sort)):
    	if visited[i]==0 and sort[i][3] == root_index:
    		tup1 = (sort[root_index-1][1], sort[root_index-1][2], root_index-1)
    		tup2 = sort[i][4]
    		tup3 = (sort[i][1],sort[i][2],i)
    		dep_tree.append((tup1,tup2,tup3))
    		visited[i] = 1

    for i in range(len(sort)):
    	if visited[i]==0:
    		tup1 = (sort[int(sort[i][3])-1][1], sort[int(sort[i][3])-1][2],int(sort[i][3])-1)
    		tup2 = sort[i][4]
    		tup3 = (sort[i][1],sort[i][2],i)
    		dep_tree.append((tup1,tup2,tup3))
    		visited[i] = 1

    return dep_tree

def read_sick(filename):
    i = 1
    with open('../data/' + filename, 'r') as inp:
        with open('../data/sick_train.txt', 'w') as train_file:
            with open('../data/sick_test.txt', 'w') as test_file:
                for line in inp:
                    arr = line.split('\t')
                    premise = arr[1]
                    hypothesis = arr[2]
                    label = arr[3]
                    output = premise + "\t" + hypothesis + "\t" + label + "\n"
                    if(i % 5 == 0):
                        test_file.write(output)
                    else:
                        train_file.write(output)
                    i = i + 1


def prepare_sick(filename):
    with open('../data/' + filename, 'r') as data_file:
        for line in data_file:
            arr = line.split('\t')
            a = arr[0]
            b = arr[1]
            c = arr[2]
            dep_tree = find_deptree(b)
            for ele in dep_tree:
                print(ele,',')
            return

# read_sick('SICK.txt')
prepare_sick('sick_train.txt')
