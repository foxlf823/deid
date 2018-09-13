from seqmodel import SeqModel
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import torch
import time
import random
import numpy as np


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    with torch.no_grad(): # feili, compatible with 0.4
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        chars = [sent[1] for sent in input_batch_list]
        labels = [sent[2] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(map(len, words))
        max_seq_len = word_seq_lengths.max()
        word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()

        mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen.item())

        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]


        label_seq_tensor = label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]
        ### deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len.item()-len(chars[idx])) for idx in range(len(chars))]
        length_list = [map(len, pad_char) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len.item(),-1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len.item(),)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        if gpu:
            word_seq_tensor = word_seq_tensor.cuda()

            word_seq_lengths = word_seq_lengths.cuda()
            word_seq_recover = word_seq_recover.cuda()
            label_seq_tensor = label_seq_tensor.cuda()
            char_seq_tensor = char_seq_tensor.cuda()
            char_seq_recover = char_seq_recover.cuda()
            mask = mask.cuda()
        return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data, opt):
    model = SeqModel(data, opt)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)

    best_dev = -10

    for idx in range(opt.iter):
        epoch_start = time.time()

        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()
        batch_size = opt.batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):

            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, opt.gpu)

            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char,
                                                          batch_charlen, batch_charrecover, batch_label, mask)

            loss.backward()
            optimizer.step()
            model.zero_grad()

        epoch_finish = time.time()
        print("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))


