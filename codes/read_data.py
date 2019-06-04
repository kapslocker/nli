from pycorenlp import StanfordCoreNLP
import subprocess
import pickle
import requests
import json
import time

def unitoStr(word):
	if( type(word) != str):
		x = uni.normalize('NFKD', word).encode('ascii','ignore')
	else:
		x = word
	return x

def find_deptree(nlp, plaintext):
	url = "http://localhost:9000/?properties={\"annotators\":\"depparse\", \"outputFormat\":\"json\"}"
	response = requests.post(url, data=plaintext)
	a = response.content
	output = json.loads(a)
	# cmd = "wget --post-data \'" + plaintext + "\' \'localhost:9000/?properties={\"annotators\":\"depparse\", \"outputFormat\":\"json\"}\' -O -"
	# print(cmd)
	# os.system(cmd)
	# output = nlp.annotate(plaintext, properties={'annotators': 'depparse', 'outputFormat': 'json'})
	# print(output)
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
			dep_tree.append( [["ROOT","ROOT",-1.1],"ROOT",[sort[i][1],sort[i][2],i]] )
	for i in range(len(sort)):
		if visited[i]==0 and sort[i][3] == root_index:
			tup1 = [sort[root_index-1][1], sort[root_index-1][2], root_index-1]
			tup2 = sort[i][4]
			tup3 = [sort[i][1],sort[i][2],i]
			dep_tree.append([tup1,tup2,tup3])
			visited[i] = 1

	for i in range(len(sort)):
		if visited[i]==0:
			tup1 = [sort[int(sort[i][3])-1][1], sort[int(sort[i][3])-1][2],int(sort[i][3])-1]
			tup2 = sort[i][4]
			tup3 = [sort[i][1],sort[i][2],i]
			dep_tree.append([tup1,tup2,tup3])
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


def read_sizes(filename):
	sizes = 0
	maxlen = 0
	count = 0
	set_max_len = 20
	long_sent = []
	with open('../data/' + filename, 'r') as data_file:
		for line in data_file:
			arr = line.split('\t')
			a = arr[0].split(' ')
			b = arr[1].split(' ')
			c = arr[2]
			if(len(a) > set_max_len):
				count += 1
			if(len(b) > set_max_len):
				count += 1
			if len(a) > maxlen:
				maxlen = len(a)
				long_sent = a
			if len(b) > maxlen:
				maxlen = len(b)
				long_sent = b
	print(maxlen, long_sent, count)

def build_dict(filename):
	# Store dict {word: idx} in memory
	i = 0
	j = 0
	idx = dict()
	with open('../data/' + filename, 'r') as datafile:
		for line in datafile:
			arr = line.split('\t')
			sent_a = arr[0].split(' ')
			sent_b = arr[1].split(' ')
			for word in sent_a:
				if word not in idx:
					idx[word] = i
					i = i + 1
			for word in sent_b:
				if word not in idx:
					idx[word] = i
					i = i + 1
	with open('vocab.pkl', 'wb') as f:
		pickle.dump(idx, f, pickle.HIGHEST_PROTOCOL)
	print('VOCAB_SIZE', i)
	return i


def print_sick_trees(nlp, filename):
	count = 0
	with open('../data/' + filename.split('.')[0] + '_deptree.txt', 'w') as outfile:
		with open('../data/' + filename, 'r') as datafile:
			for line in datafile:
				if(count % 500 == 0):
					# kill and restart server
					cmd1 =  "wget \"localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`\" -O -"
					direct = '../data/corenlp/stanford-corenlp-full-2016-10-31'
					p = subprocess.Popen([cmd1], shell=True)
					p.wait()
					cmd2 = "java -Xmx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 50000000"
					p = subprocess.Popen(cmd2, cwd=direct, shell=True)
					time.sleep(10)
				# print(count)
				count += 1

				arr = line.split('\t')
				sent_a = arr[0]
				sent_b = arr[1]
				label = arr[2]
				dep_a = find_deptree(nlp,sent_a)
				# return
				dep_b = find_deptree(nlp,sent_b)
				out_line = ""
				for ele in dep_a:
					out_line += ele[0][0]+','+ele[0][1]+','+str(ele[0][2])+','+ele[1]+','+ele[2][0]+','+ele[2][1]+','+str(ele[2][2])+' '
				out_line += '\t'
				for ele in dep_b:
					out_line += ele[0][0]+','+ele[0][1]+','+str(ele[0][2])+','+ele[1]+','+ele[2][0]+','+ele[2][1]+','+str(ele[2][2])+' '
				out_line += '\t'
				out_line += label + '\n'

				outfile.write(out_line)



# read_sick('SICK.txt')
# prepare_sick('sick_train.txt')

if __name__ == '__main__':
	prepare_sick('sick_train.txt')
	# nlp = StanfordCoreNLP("http://127.0.0.1:9000")
	# print_sick_trees(nlp, 'sick_train.txt')
