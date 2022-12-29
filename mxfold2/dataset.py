from itertools import groupby
from torch.utils.data import Dataset
import torch
import math
import _pickle as cPickle
import numpy as np
import os

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}

def encoding2seq(arr):
	seq = list()
	for arr_row in list(arr):
		if sum(arr_row)==0:
			seq.append('.')
		else:
			seq.append(char_dict[np.argmax(arr_row)])
	return ''.join(seq)


class RNASSDataGenerator(object):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        #p = Pool()   # CJY
        data_dir = './data/SPOT_RNA_500'
        # Load the current split
        print("data:", os.path.join(data_dir, '%s.pickle' % self.split))
        with open(os.path.join(data_dir, '%s.pickle' % self.split), 'rb') as f:
            self.data = cPickle.load(f)
        self.data_x = np.array([instance.seq for instance in self.data])
        self.data_y = np.array([instance.ss_label for instance in self.data])
        self.pairs = np.array([instance.pairs for instance in self.data], dtype=object)
        self.seq_length = np.array([instance.length for instance in self.data])
        self.len = len(self.data)
        self.name = np.array([instance.name for instance in self.data])
        #self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq = list(map(encoding2seq, self.data_x))  # CJY
        self.seq_list = np.array([instance.seq_list for instance in self.data])
        self.seq_max_len = len(self.data_x[0])
        self.fm_embedding = np.array([instance.embedding for instance in self.data])   #np.array([instance.embedding for instance in self.data])
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp:bp + batch_size]
        batch_y = self.data_y[bp:bp + batch_size]
        batch_seq_len = self.seq_length[bp:bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact
    
    def pairs2pairs(self, pairs, length):
        new_pairs = np.zeros(length+1, dtype=int)
        for pair in pairs:
            new_pairs[pair[0]+1] = pair[1]+1
        return new_pairs


    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[index]
        # data_seq = self.seq[index] # padding to 500 length
        data_seq = self.seq_list[index] # no padding
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        print(data_seq, data_pair)
        data_pair = self.pairs2pairs(data_pair, len(data_seq))
        fm_embedding = self.fm_embedding[index]
        # contact= self.pairs2map(data_pair)
        # matrix_rep = np.zeros(contact.shape)
        name = self.name[index]
        # print(name, data_seq, torch.Tensor(data_pair).shape, fm_embedding.shape)
        return name, data_seq, torch.Tensor(data_pair).type(torch.IntTensor)


    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq

    def get_one_sample_cdp(self, index):
        data_seq = self.data_x[index]
        data_label = self.data_y[index]

        return data_seq, data_label

# using torch data loader to parallel and speed up the data load process
class mxfold2Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample(index)

class FastaDataset(Dataset):
    def __init__(self, fasta):
        it = self.fasta_iter(fasta)
        try:
            self.data = list(it)
        except RuntimeError:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq, torch.tensor([]))


class BPseqDataset(Dataset):
    def __init__(self, bpseq_list):
        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l)==1:
                    self.data.append(self.read(l[0]))
                elif len(l)==2:
                    self.data.append(self.read_pdb(l[0], l[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(filename) as f:
            structure_is_known = True
            p = [0]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    if len(l) == 3:
                        if not structure_is_known:
                            raise('invalid format: {}'.format(filename))
                        idx, c, pair = l
                        pos = 'x.<>|'.find(pair)
                        if pos >= 0:
                            idx, pair = int(idx), -pos
                        else:
                            idx, pair = int(idx), int(pair)
                        s.append(c)
                        p.append(pair)
                    elif len(l) == 4:
                        structure_is_known = False
                        idx, c, nll_unpaired, nll_paired = l
                        s.append(c)
                        nll_unpaired = math.nan if nll_unpaired=='-' else float(nll_unpaired)
                        nll_paired = math.nan if nll_paired=='-' else float(nll_paired)
                        p.append([nll_unpaired, nll_paired])
                    else:
                        raise('invalid format: {}'.format(filename))
        
        if structure_is_known:
            seq = ''.join(s)
            return (filename, seq, torch.tensor(p))
        else:
            seq = ''.join(s)
            p.pop(0)
            return (filename, seq, torch.tensor(p))

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)

    def read_pdb(self, seq_filename, label_filename):
        it = self.fasta_iter(seq_filename)
        h, seq = next(it)

        p = []
        with open(label_filename) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                    p.append([int(l[0]), int(l[1])])

        return (h, seq, torch.tensor(p))