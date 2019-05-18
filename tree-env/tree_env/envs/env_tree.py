import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import defaultdict


class TreeEnv(gym.Env):
    metadata = {'render-modes': ['human']}

    def __init__(self):
        self.premise_tree = []
        self.hypothesis_tree = []
        self.label = "NEUTRAL"
        self.TOTAL_TIME_STEPS = 100
        self.available_actions = []
        self.action_space = spaces.Discrete(0)
        # Observation is the distance between current tree and hypothesis_tree.
        low = -50
        high = 50
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.current_episode = -1
        self.current_step = -1
        self.action_memory = [] # Storing steps here to debug
        # Call setParams to init trees and label.
        pass

    def setParams(self, tree1, tree2, label="NONE"):
        # Action space changes after each step.
        self.premise_tree = tree1
        self.hypothesis_tree = tree2
        if(label != "NONE"):
            self.label = label
        self.available_actions = get_available_actions(tree1, tree2)
        self.action_space = spaces.Discrete(len(self.available_actions))

    def step(self, action):
        # need to return obs(obj), reward(float), episode_over(bool), info(dict)
        self.current_step += 1
        self._take_action(action)
        reward = self._get_reward()
        obs = self._finddist(self.premise_tree, self.hypothesis_tree)
        pass

    def _take_action(action):
        pass

    def _get_reward():
        return 0

    def reset(self):
        pass


    def _finddist(deptree_curr, deptree_target):
        k_curr_target = _findsim(deptree_curr, deptree_target)
	    k_target_target = _findsim(deptree_target, deptree_target)
        k_curr_curr = _findsim(deptree_curr, deptree_curr)
        return  1 - (k_curr_target)/(pow(k_curr_curr*k_target_target,0.5))

    def _findsim(deptree_curr, deptree_target):
    	acc = 0
    	simdict = {}
    	for node1 in deptree_curr:
    		for node2 in deptree_target:
    			delta_nodes = _delta(node1, node2, deptree_curr, deptree_target)
    			acc += delta_nodes
    	return acc

    def delta(node1, node2, deptree_curr, deptree_target):
    	childlistnode1 = []
    	childlistnode2 = []
    	for ele in deptree_curr:
    		if ele[0][2] == node1[2][2]:
    			childlistnode1.append(ele)

    	for ele in deptree_target:
    		if ele[0][2] == node2[2][2]:
    			childlistnode2.append(ele)
    	childlistnode1 = sorted(childlistnode1, key = lambda element : element[2][2])
    	childlistnode2 = sorted(childlistnode2, key = lambda element : element[2][2])
    	sum_child = 0

    	for l in range(1, min(len(childlistnode1), len(childlistnode2)) + 1):
    		for i1 in range(0, len(childlistnode1) - l + 1):
    			j1 = l + i1 - 1
    			for i2 in range(0, len(childlistnode2) - l + 1):
    				j2 = l + i2 - 1
    				product = 1
    				for k in range(l):
    					product *= delta(childlistnode1[i1 + k],
                        childlistnode2[i2 + k],
                        deptree_curr,
                        deptree_target)
    				sum_child += product
    	return mu*(pow(Lambda,2)*nodesim(node1, node2) + sum_child)

    def render(self, mode='human', close=False):
        return

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


if __name__ == '__main__':
