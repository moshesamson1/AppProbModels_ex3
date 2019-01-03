import sys


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
    articles = f_lines[2::2]
    topics = f_lines[0::2]
    words = [word for line in f_lines for word in line.split() if word != '\r\r\n']
    z = list(zip(articles, topics))
    words_dict = list_to_dictionary(words)
    filtered_words = get_filtered_words(words_dict)

    print("!")


def list_to_dictionary(my_list):
    voc = dict()

    for element in my_list:
        voc[element] = voc.setdefault(element, 0) + 1
    return voc


if __name__ == "__main__":
    main(sys.argv)
