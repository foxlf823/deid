import codecs
from alphabet import Alphabet


def loadData(file_path):
    file = codecs.open(file_path, 'r', 'UTF-8')
    data = []
    sentence = []
    for line in file:
        columns = line.strip().split()
        if len(columns) == 0:
            data.append(sentence)
            sentence = []
            continue

        token = {}
        token['text'] = columns[0]
        token['doc'] = columns[1]
        token['start'] = int(columns[2])
        token['end'] = int(columns[3])
        token['label'] = columns[5]
        sentence.append(token)

    file.close()
    return data

def read_instance(data, word_alphabet, char_alphabet, label_alphabet):

    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for sentence in data:
        for token in sentence:
            words.append(token['text'])
            word_Ids.append(word_alphabet.get_index(token['text']))
            labels.append(token['label'])
            label_Ids.append(label_alphabet.get_index(token['label']))
            char_list = []
            char_Id = []
            for char in token['text']:
                char_list.append(char)
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        instence_texts.append([words, chars, labels])
        instence_Ids.append([word_Ids, char_Ids, label_Ids])
        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        label_Ids = []

    return instence_texts, instence_Ids



class Data:
    def __init__(self):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)

        self.train_texts = None
        self.train_Ids = None
        self.dev_texts = None
        self.dev_Ids = None
        self.test_texts = None
        self.test_Ids = None

    def build_alphabet(self, data):
        for sentence in data:
            for token in sentence:
                self.word_alphabet.add(token['text'])
                self.label_alphabet.add(token['label'])
                for char in token['text']:
                    self.char_alphabet.add(char)

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()








