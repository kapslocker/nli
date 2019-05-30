import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import defaultdict

class TreeEnv(gym.Env):
    metadata = {'render-modes': ['human']}

    def __init__(self):
        self.MAX_SENTENCE_SIZE = 20
        self.MAX_NUM_STEPS = 100
        self.action_name_map = {'DECISION': 0, 'INSERT_CHILD': 1, 'INSERT_PARENT': 2, 'DELETE': 3, 'RELABEL_NODE': 4}
        self.name_action_map = {0: 'DECISION', 1: 'INSERT_CHILD', 2: 'INSERT_PARENT', 3: 'DELETE',4: 'RELABEL_NODE'}
        self.action_space = spaces.Discrete(5 * self.MAX_SENTENCE_SIZE * self.MAX_SENTENCE_SIZE)
        self.current_step = -1
        self.done = False

    def setParams(self, tree1, tree2, label):
        self.premise_tree = tree1
        self.hypothesis_tree = tree2
        self.label = label

    def reset(self):
        self.current_step = -1
        self.done = False
        self.premise_tree = []
        self.hypothesis_tree = []

    def step(self, action):
        # need to return obs(obj), reward(float), episode_over(bool), info(dict)
        # obs -> next state after step.
        if self.done:
            return self.premise_tree, 0, self.done, self.premise_tree
        self.current_step += 1
        tree_copy = self.premise_tree
        is_correct = self._take_action(action)
        if is_correct == 2 or is_correct == 3 or self.current_step > self.MAX_NUM_STEPS:
            self.done = True
        reward = self._get_reward(tree_copy, is_correct)
        obs = self.premise_tree
        return obs, reward, self.done, obs

    def _take_action(self, action):
        ''' Returns result of action done.
            0 -> invalid action.
            1 -> valid action, correct execution.
            2 -> valid action, correct decision taken. (for decision actions)
            3 -> valid action, incorrect  decision taken. (for decision actions)
        '''
        available_actions = self.get_available_actions()
        list_of_actions = self.encode_tuples(available_actions)
        if action not in list_of_actions:
            return 0
        temp = self.decode_action(action)
        action_type = self.name_action_map[temp[0]]
        source_node_id = temp[1]
        dest_node_id = temp[2]
        if action_type == 'DECISION':
            ans = self._decision_action(dest_node_id)
            if ans > 0:
                return 2
            return 3
        elif action_type == 'INSERT_CHILD':
            self._insert_child_action(source_node_id, dest_node_id)
        elif action_type == 'INSERT_PARENT':
            self._insert_parent_action(source_node_id, dest_node_id)
        elif action_type == 'DELETE':
            self._delete_node_action(dest_node_id)  # it is not dest node id, but it is the last parameter.
        elif action_type == 'RELABEL_NODE':
            self._relabel_node_action(source_node_id, dest_node_id)
        return 1

    def _get_reward(self, old_tree, is_correct):
        '''
        Handle rewards here.
        1. invalid action taken => -10
        2. tree edit action taken => +1 (went closer) / -1 (went further)
        3. correct decision action taken => +20
        4. incorrect decision action taken => -20
        '''
        if is_correct == 0:
            return -10
        elif is_correct == 1:
            # prev_dist = self._finddist(old_tree, self.hypothesis_tree)
            # curr_dist = self._finddist(self.premise_tree, self.hypothesis_tree)
            # if curr_dist < prev_dist:
            #     return 1
            # else:
            #     return -1
            return -1
        elif is_correct == 2:
            return 20
        else:
            return -20

    def encode_tuples(self, list_of_tuples):
        n_categories = len(self.action_name_map)
        encoded_list = []
        for item in list_of_tuples:
            action_type = self.action_name_map[item[0]]
            param_1 = item[1]
            param_2 = item[2]
            value = action_type * self.MAX_SENTENCE_SIZE * self.MAX_SENTENCE_SIZE
            value = value + param_1 * self.MAX_SENTENCE_SIZE
            value = value + param_2
            encoded_list.append(value)
        return encoded_list

    def decode_action(self, action):
        ''' action: Integer
            Decode to 3-tuple [ACTION, PARAM_1, PARAM_2]
        '''
        a = [0,0,0]
        a[2] = action % self.MAX_SENTENCE_SIZE
        action = action // self.MAX_SENTENCE_SIZE
        a[1] = action % self.MAX_SENTENCE_SIZE
        action = action // self.MAX_SENTENCE_SIZE
        a[0] = action
        return a


    def get_available_actions(self):
        '''
        Return tuple based list of available actions.
        [a,b,c] : a -> type of action.
                b-> param 1
                c-> param 2
        '''
        destination_tree = self.hypothesis_tree
        source_tree = self.premise_tree
        actions = []
        actions.append(['DECISION', 0, 0])  # Entailment
        actions.append(['DECISION', 0, 1])  # Neutral
        actions.append(['DECISION', 0, 2])  # Contradiction
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
                actions.append(['INSERT_CHILD', parent[2][2], node_data[2]])

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
                actions.append(['INSERT_PARENT', elem[2][2], node_data[2]])

        # Delete actions.
        for node in source_tree:
            node_data = node[2]
            parent_data = node[0]
            edge_label = node[1]
            if node_data[0] not in words_to_delete:
                continue
            # Current word is extra, and has to be deleted.
            actions.append(['DELETE', 0, node_data[2]])

        # Relabel node actions.
        for node in destination_tree:
            node_data = node[2]
            if node_data[0] not in words_to_insert:
                continue
            for elem in source_tree:
                if elem[2][0] not in words_to_delete:
                    continue
                actions.append(['RELABEL_NODE', elem[2][2], node_data[2]])

        # Relabel edge actions.
        # edge_map = defaultdict(int)
        # for node in destination_tree:
        #     edge_map[node[1]] += 1
        #
        # for node in source_tree:
        #     edge_map[node[2]] -= 1
        #
        # labels_to_remove = []
        # labels_to_add = []
        # for (key, value) in edge_map.items():
        #     if value < 0:
        #         labels_to_remove.append(key)
        #     elif value > 0:
        #         labels_to_add.append(key)
        #
        # for node in source_tree:
        #     edge_label = node[1]
        #     if edge_label in labels_to_remove:
        #         for label in labels_to_add:
        #             actions.append(['RELABEL_EDGE', node, label])
        return actions


    def _insert_child_action(self, source_node_id, dest_node_id):
        ''' Insert dest_node_id as a child for source_node_id '''
        for node in self.premise_tree:
            source_node_data = node[2]
            if source_node_data[2] != source_node_id:
                continue
            for dest_node in self.hypothesis_tree:
                dest_node_data = dest_node[2]
                edge_label = dest_node[1]
                if dest_node_data[2] != dest_node_id:
                    continue

                new_node_id = 0
                if dest_node[0][2] > dest_node[2][2]:
                    # insert as leftmost child on subtree of source_node_data.
                    left_idx = 100
                    par_idx = source_node_data[2]
                    for temp in self.premise_tree:
                        if temp[0][2] == par_idx:
                            left_idx = min(left_idx, temp[2][2])
                    # insert at left_idx and shift all elements in premise tree ahead by 1.
                    # make space now.
                    for i in range(len(self.premise_tree)):
                        if self.premise_tree[i][0][2] >= left_idx:
                            self.premise_tree[i][0][2] += 1
                        if self.premise_tree[i][2][2] >= left_idx:
                            self.premise_tree[i][2][2] += 1
                    if par_idx >= left_idx:
                        par_idx += 1
                    self.premise_tree.append([[source_node_data[0], source_node_data[1], par_idx],
                                              edge_label,
                                              [dest_node_data[0], dest_node_data[1], left_idx]])
                    return
                else:
                    # insert as rightmost child in subtree of source_node_data.
                    # same as for leftmost child.
                    right_idx = -1
                    par_idx = source_node_data[2]
                    for temp in self.premise_tree:
                        if temp[0][2] == par_idx:
                            right_idx = max(right_idx, temp[2][2])
                    for i in range(len(self.premise_tree)):
                        if self.premise_tree[i][0][2] >= right_idx:
                            self.premise_tree[i][0][2] += 1
                        if self.premise_tree[i][2][2] >= right_idx:
                            self.premise_tree[i][2][2] += 1
                    if par_idx >= right_idx:
                        par_idx += 1
                    self.premise_tree.append([[source_node_data[0], source_node_data[1], par_idx],
                                                edge_label,
                                                [dest_node_data[0], dest_node_data[1], right_idx]])

    def _insert_parent_action(self, source_node_id, dest_node_id):
        ''' Insert dest_node_id as parent for source_node_id
            1. Find parent for source_node_id, break this edge.
            2. Insert dest_node_id as child of parent.
            3. Insert source_node_id as child for dest_node_id.'''
        for node in self.premise_tree:
            source_node_data = node[2]
            if source_node_data[2] != source_node_id:
                continue
            for dest_node in self.hypothesis_tree:
                dest_node_data = dest_node[2]
                edge_label = dest_node[1]
                if dest_node_data[2] != dest_node_id:
                    continue

        for i in range(len(self.premise_tree)):
            node = self.premise_tree[i]
            if node[2][2] != source_node_id:
                continue
            for dest_node in self.hypothesis_tree:
                dest_node_data = dest_node[2]
                edge_label = dest_node[1]
                if dest_node_data[2] != dest_node_id:
                    continue
                # edit current edge to have new node in place of child node.
                if dest_node[0][2] > dest_node[2][2]:
                    # insert on left side.
                    left_idx = 100
                    par_idx = node[0][2]
                    for temp in self.premise_tree:
                        if temp[0][2] == par_idx:
                            left_idx = min(left_idx, temp[2][2])
                    for j in range(len(self.premise_tree)):
                        if self.premise_tree[j][0][2] >= left_idx:
                            self.premise_tree[j][0][2] += 1
                        if self.premise_tree[j][2][2] >= left_idx:
                            self.premise_tree[j][2][2] += 1
                    if par_idx >= left_idx:
                        par_idx += 1
                    new_par_idx = left_idx
                    child_idx = node[2][2]
                    if node[2][2] >= left_idx:
                        child_idx += 1
                    old_par_idx = par_idx
                    self.premise_tree[i] = [[dest_node_data[0], dest_node_data[1], new_par_idx], node[1], [node[2][0], node[2][1], child_idx]]
                    self.premise_tree.append([[node[0][0], node[0][1], old_par_idx], edge_label, [dest_node_data[0], dest_node_data[1], new_par_idx]])

                if dest_node[0][2] < dest_node[2][2]:
                    # insert on right side
                    right_idx = -1
                    par_idx = node[0][2]
                    for temp in self.premise_tree:
                        if temp[0][2] == par_idx:
                            right_idx = max(right_idx, temp[2][2])
                    for j in range(len(self.premise_tree)):
                        if self.premise_tree[j][0][2] >= right_idx:
                            self.premise_tree[j][0][2] += 1
                        if self.premise_tree[j][2][2] >= right_idx:
                            self.premise_tree[j][2][2] += 1
                    if par_idx >= right_idx:
                        par_idx += 1
                    new_par_idx = right_idx
                    child_idx = node[2][2]
                    if node[2][2] >= right_idx:
                        child_idx += 1
                    old_par_idx = par_idx
                    self.premise_tree[i] = [[dest_node_data[0], dest_node_data[1], new_par_idx], node[1], [node[2][0], node[2][1], child_idx]]
                    self.premise_tree.append([[node[0][0], node[0][1], old_par_idx], edge_label, [dest_node_data[0], dest_node_data[1], new_par_idx]])

    def _decision_action(self, action_id):
        decisions = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
        if decisions[action_id] == self.label:
            return True
        return False

    def _relabel_node_action(self, source_node_id, dest_node_id):
        ''' Relabel source_node_id to dest_node_id.'''
        for i in range(len(self.premise_tree)):
            node = self.premise_tree[i]
            if node[2][2] != source_node_id:
                continue
            for dest_node in self.hypothesis_tree:
                if dest_node[2][2] != dest_node_id:
                    continue
                self.premise_tree[i][1] = dest_node[1]
                self.premise_tree[i][2][0] = dest_node[2][0]
                self.premise_tree[i][2][1] = dest_node[2][1]

    def _delete_node_action(self, source_node_id):
        ''' Delete Node with id = source_node_id.'''
        for i in range(len(self.premise_tree)):
            node = self.premise_tree[i]
            if node[2][2] == source_node_id:
                parent_node = node[0]
                del self.premise_tree[i]
                break
        # Add links between the deleted node's children and its parent.
        for i in range(len(self.premise_tree)):
            if self.premise_tree[i][0][2] == source_node_id:
                self.premise_tree[i][0][0] = parent_node[0]
                self.premise_tree[i][0][1] = parent_node[1]
                self.premise_tree[i][0][2] = parent_node[2]
        # move nodes backward to cover freed space.
        for i in range(len(self.premise_tree)):
            if self.premise_tree[i][0][2] >= source_node_id:
                self.premise_tree[i][0][2] -= 1
            if self.premise_tree[i][2][2] >= source_node_id:
                self.premise_tree[i][2][2] -= 1

    def render(self, mode='human', close=False):
        print("==========")
        for ele in self.premise_tree:
            print(ele)
        print("----------")
        for ele in self.hypothesis_tree:
            print(ele)
        print("==========")


if __name__ == '__main__':
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
    label = 'ENTAILMENT'
    tree[:] = [[list(node[0]),node[1],list(node[2])] for node in tree]
    tree1[:] = [[list(node[0]),node[1],list(node[2])] for node in tree1]
    print(tree)
    env = TreeEnv()
    print(env.action_space.n)
    env.setParams(tree, tree1, label)
    print(env.done)
    a,b,c,d = env.step(503)
    print(b,c, "------------")
    print(a)
    # a,b,c,d = env.step(1)
    # print(b,c, "------------")
    # env.reset()
    # print(env.hypothesis_tree)
