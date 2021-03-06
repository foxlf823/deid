import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=1, help='1-train, 2-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-train_file', default='./sample.txt')
parser.add_argument('-dev_file', default='./sample.txt')
parser.add_argument('-test_file', default='./sample.txt')
parser.add_argument('-output', default='./output')
parser.add_argument('-iter', type=int, default=100)
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-tune_wordemb', action='store_true', default=False)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-word_emb_file', default='/Users/feili/project/emb_100_for_debug.bin')
parser.add_argument('-word_emb_dim', type=int, default=100)
parser.add_argument('-hidden_dim', type=int, default=100)
parser.add_argument('-char_emb_dim', type=int, default=24)
parser.add_argument('-char_hidden_dim', type=int, default=24)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-l2', type=float, default=1e-8)
parser.add_argument('-nbest', type=int, default=0)
parser.add_argument('-patience', type=int, default=10)
parser.add_argument('-gradient_clip', type=float, default=5.0)


opt = parser.parse_args()

