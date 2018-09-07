
import random
import numpy as np
import torch

from options import opt
import data

print(opt)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

d = data.Data()

d.train_data = data.loadData(opt.train_file)
d.dev_data = data.loadData(opt.dev_file)
d.test_data = data.loadData(opt.test_file)


d.build_alphabet(d.train_data)
d.build_alphabet(d.dev_data)
d.build_alphabet(d.test_data)
d.fix_alphabet()

d.train_texts, d.train_Ids = data.read_instance(d.train_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)
d.dev_texts, d.dev_Ids = data.read_instance(d.dev_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)
d.test_texts, d.test_Ids = data.read_instance(d.test_data, d.word_alphabet, d.char_alphabet, d.label_alphabet)
pass