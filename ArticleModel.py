import numpy as np

class ArticleModel:
    def __init__(self, words, topics):
        self.num_topics = 9
        self.words = words
        self.real_topics = topics
        self.topic_idx = -1
        self.Wti = [0]*self.num_topics
        self.words_appearances_dict = dict()
        self.Mnt = 0
        self.zValues = [0]*self.num_topics

    def create_ntk(self, frequent_words):
        for word in frequent_words:
            self.words_appearances_dict[word] = 0
            if word in self.words:
                self.Mnt += 1
                self.words_appearances_dict[word] += 1


    def set_topic_idx(self, topic_idx):
        self.topic_idx = topic_idx
        self.Wti[self.topic_idx-1] = 1

    def get_Wti(self, idx):
        return self.Wti[idx]

    def set_Wti(self, values):
        self.Wti = values

    def get_Mnt(self):
        return self.Mnt

    def get_Ntk(self, idx):
        return self.words_appearances_dict[idx]

    def get_Ntk_full(self):
        return self.words_appearances_dict

    def get_zValues(self):
        return self.zValues

    def set_zValues(self, values):
        self.zValues = values