import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: size of word vectors (default 300)
        :param hidden_size: LSTM hidden size
        """
        super(ChildSumTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TreeLSTM gates
        self.ioux = nn.Linear(self.input_size, 3 * self.hidden_size)
        self.iouh = nn.Linear(self.hidden_size, 3 * self.hidden_size)

        self.fx = nn.Linear(self.input_size, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

    def node_forward(self, inputs, child_c, child_h):
        """"""

        # sum over hidden states of child nodes
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        # TreeLSTM gates computation
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )

        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        """"""
        # iterate over child nodes
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:  # leaf node
            child_c = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
            child_h = Variable(inputs[0].data.new(1, self.hidden_size).fill_(0.))
        else:  # internal node
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)  # tree.idx from 0

        return tree.state

class NLITreeLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, mem_size, hidden_size, num_classes,
                 freeze_embed, edge_vocab_size=100, edge_size=20, dropout=None):
        super(NLITreeLSTM, self).__init__()
        self.embs = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if freeze_embed:
            self.embs.weight.requires_grad = False

        self.edge_embs = nn.Embedding(edge_vocab_size, edge_size)
        self.dropout = dropout

        self.treelstm = ChildSumTreeLSTM(input_size=emb_size, hidden_size=mem_size)

    def forward(self, ltree, linputs, rtree, rinputs, ledges=None, redges=None):
        """"""
        # compute word vectors
        linputs = self.embs(linputs)
        rinputs = self.embs(rinputs)

        if ledges is not None and redges is not None:
            if self.model != 'full':
                ledges = self.edge_embs(ledges)
                redges = self.edge_embs(redges)
            lstate, lhidden = self.treelstm(ltree, linputs, ledges)
            rstate, rhidden = self.treelstm(rtree, rinputs, redges)
        else:
            lstate, lhidden = self.treelstm(ltree, linputs)
            rstate, rhidden = self.treelstm(rtree, rinputs)

        # output = self.classifier(lstate, rstate)
    pass
