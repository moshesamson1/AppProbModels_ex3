import sys
from ClusteringModel import *
from ArticleModel import *

def create_dict(words):
    return dict()


def get_filtered_words(words_dict):
    return [i for i in words_dict.keys() if int(words_dict[i]) > 3]


def main(args):
    filename = ""
    if len(args) == 2:
        filename = args[1]
    f = open(filename, 'r')
    f_lines = f.readlines()
    articles = f_lines[1::2]
    topics_per_article = [(line[1:-4]).split()[2:] for line in f_lines[0::2]]
    article_models = [ArticleModel(articles[i].split(), topics_per_article[i]) for i in xrange(len(articles))]
    words = [word for line in f_lines for word in line.split() if word != '\r\r\n']
    possible_topics = set([topic for topic_line in topics_per_article for topic in topic_line])

    words_dict = list_to_dictionary(words)
    frequent_words = get_filtered_words(words_dict)

    model = ClusteringModel(article_models, frequent_words)
    model.init_parameters()
    model.cluster()

    print("there are %d articles, and %d different possible topics" % (len(articles), len(possible_topics)))
    print possible_topics


def list_to_dictionary(my_list):
    voc = dict()

    for element in my_list:
        voc[element] = voc.setdefault(element, 0) + 1
    return voc


if __name__ == "__main__":
    main(sys.argv)
