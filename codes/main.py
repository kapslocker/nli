'''
Type of actions:
1. INSERT_NODE
2. DELETE_&_MERGE
3. RELABEL_NODE
4. RELABEL_EDGE
5. PREDICT_ENTAILMENT - {NEUTRAL, ENTAILMENT, CONTRADICTION}
'''

from collections import defaultdict

def get_available_actions(source_tree, destination_tree):
    actions = []
    actions.append(['DECISION', 'ENTAILMENT'])
    actions.append(['DECISION', 'NEUTRAL'])
    actions.append(['DECISION', 'CONTRADICTION'])
    words_to_insert = []
    words_to_delete = []
    word_map = defaultdict(int)

    for node in destination_tree:
        lemma = node[2][0]
        word_map[lemma] += 1

    for node in source_tree:
        lemma = node[2][0]
        word_map[lemma] -= 1

    for (key, value) in word_map.items():
        if value > 0:
            words_to_insert.append(key)
        elif(value < 0):
            words_to_delete.append(key)

    # Insert node actions.
    for node in destination_tree:
        node_data = node[2]
        parent_data = node[0]
        edge_label = node[1]
        if node_data[0] not in words_to_insert:
            continue
        # Current node needs to be inserted.
        # It can be inserted anywhere.
        for parent in source_tree:
            actions.append(['INSERT_CHILD', parent[2], node_data, edge_label])

    # Insert parent actions.
    for node in destination_tree:
        node_data = node[2]
        parent_data = node[0]
        edge_label = node[1]
        if node_data[0] not in words_to_insert:
            continue
        for elem in source_tree:
            if elem[1] == 'ROOT':
                continue
            actions.append(['INSERT_PARENT', elem, node_data, edge_label])

    # Delete actions.
    for node in source_tree:
        node_data = node[2]
        parent_data = node[0]
        edge_label = node[1]
        if node_data[0] not in words_to_delete:
            continue
        # Current word is extra, and has to be deleted.
        actions.append(['DELETE', node_data])

    # Relabel node actions.
    for node in destination_tree:
        node_data = node[2]
        if node_data[0] not in words_to_insert:
            continue
        for elem in source_tree:
            if elem[2][0] not in words_to_delete:
                continue
            actions.append(['RELABEL_NODE', elem[2], node_data])

    # Relabel edge actions.
    edge_map = defaultdict(int)
    for node in destination_tree:
        edge_map[node[1]] += 1

    for node in source_tree:
        edge_map[node[2]] -= 1

    labels_to_remove = []
    labels_to_add = []
    for (key, value) in edge_map.items():
        if value < 0:
            labels_to_remove.append(key)
        elif value > 0:
            labels_to_add.append(key)

    for node in source_tree:
        edge_label = node[1]
        if edge_label in labels_to_remove:
            for label in labels_to_add:
                actions.append(['RELABEL_EDGE', node, label])

    return actions

    

tree = [(('ROOT', 'ROOT', -1.1), 'ROOT', ('playing', 'VBG', 5)) ,
(('playing', 'VBG', 5), 'nsubj', ('group', 'NN', 1)) ,
(('playing', 'VBG', 5), 'aux', ('is', 'VBZ', 4)) ,
(('playing', 'VBG', 5), 'nmod', ('yard', 'NN', 8)) ,
(('playing', 'VBG', 5), 'dep', ('standing', 'VBG', 14)) ,
(('group', 'NN', 1), 'det', ('A', 'DT', 0)) ,
(('kids', 'NNS', 3), 'case', ('of', 'IN', 2)) ,
(('group', 'NN', 1), 'nmod', ('kids', 'NNS', 3)) ,
(('yard', 'NN', 8), 'case', ('in', 'IN', 6)) ,
(('yard', 'NN', 8), 'det', ('a', 'DT', 7)) ,
(('yard', 'NN', 8), 'cc', ('and', 'CC', 9)) ,
(('man', 'NN', 12), 'det', ('an', 'DT', 10)) ,
(('man', 'NN', 12), 'amod', ('old', 'JJ', 11)) ,
(('yard', 'NN', 8), 'conj', ('man', 'NN', 12)) ,
(('standing', 'VBG', 14), 'aux', ('is', 'VBZ', 13)) ,
(('background', 'NN', 17), 'case', ('in', 'IN', 15)) ,
(('background', 'NN', 17), 'det', ('the', 'DT', 16)) ,
(('standing', 'VBG', 14), 'nmod', ('background', 'NN', 17))]

tree1 = [(('ROOT', 'ROOT', -1.1), 'ROOT', ('playing', 'VBG', 8)) ,
(('playing', 'VBG', 8), 'nsubj', ('group', 'NN', 1)) ,
(('playing', 'VBG', 8), 'aux', ('is', 'VBZ', 7)) ,
(('playing', 'VBG', 8), 'cc', ('and', 'CC', 9)) ,
(('playing', 'VBG', 8), 'conj', ('standing', 'VBG', 13)) ,
(('group', 'NN', 1), 'det', ('A', 'DT', 0)) ,
(('boys', 'NNS', 3), 'case', ('of', 'IN', 2)) ,
(('group', 'NN', 1), 'nmod', ('boys', 'NNS', 3)) ,
(('yard', 'NN', 6), 'case', ('in', 'IN', 4)) ,
(('yard', 'NN', 6), 'det', ('a', 'DT', 5)) ,
(('group', 'NN', 1), 'nmod', ('yard', 'NN', 6)) ,
(('man', 'NN', 11), 'det', ('a', 'DT', 10)) ,
(('standing', 'VBG', 13), 'nsubj', ('man', 'NN', 11)) ,
(('standing', 'VBG', 13), 'aux', ('is', 'VBZ', 12)) ,
(('background', 'NN', 16), 'case', ('in', 'IN', 14)) ,
(('background', 'NN', 16), 'det', ('the', 'DT', 15)) ,
(('standing', 'VBG', 13), 'nmod', ('background', 'NN', 16))]


if __name__ == '__main__':
    actions = get_available_actions(tree, tree1)
    for action in actions:
        if action[0] == 'INSERT_CHILD':
            print(action)
