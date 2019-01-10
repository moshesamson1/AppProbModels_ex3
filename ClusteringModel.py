import numpy as np
import math
import time


class ClusteringModel:
    def __init__(self, articles_models, frequent_words):
        self.eps = 0.1
        self.stop_threshold = 1
        self.num_clusters = 9
        self.articles_models = articles_models
        self.num_articles = len(self.articles_models)
        self.frequent_words = frequent_words
        self.Mv = len(frequent_words)
        self.indices = []
        self.alphas = []
        self.Piks = np.zeros((self.num_clusters, self.Mv))
        self.LAMBDA = 1.0
        self.k_value = 10

    def init_parameters(self):
        for article_idx in range(self.num_articles):
            self.articles_models[article_idx].set_topic_idx((article_idx % self.num_clusters))
            self.articles_models[article_idx].create_ntk(self.frequent_words)

        self.compute_alphas()
        self.compute_Piks()
        self.compute_z()


    def compute_W(self):
        print("Computing W... "),
        s = time.time()
        for article in self.articles_models:
            z_values = article.get_zValues()
            m = max(z_values)

            sum_values = 0.0
            for c_idx in range(self.num_clusters):
                if c_idx - m >= -self.k_value:
                    sum_values += math.exp(z_values[c_idx] - m)

            w_values_new = [0]*self.num_clusters
            for c_idx in range(self.num_clusters):
                if z_values[c_idx] - m >= -self.k_value:
                    w_values_new[c_idx] = math.exp(z_values[c_idx]-m)/sum_values

            article.set_Wti(w_values_new)
        print("done (" + str(time.time() - s) + ")")

    def compute_alphas(self):
        print("Computing Alphas... "),
        s = time.time()
        self.alphas = [(sum([article.Wti[i] for article in self.articles_models]))/float(self.num_articles)
                       for i in range(self.num_clusters)]
        self.alphas = [self.eps if x < self.eps else x for x in self.alphas]
        su = sum(self.alphas)
        self.alphas = [x/su for x in self.alphas]
        print("done (" + str(time.time() - s) + ")")


    def compute_Piks(self):
        """
        The probability of a word Wk in topic Ti
        :return:
        """
        print("Computing Pik... "),
        s = time.time()
        for c_idx in range(self.num_clusters):
            print("cluster: " + str(c_idx)),

            denominator = self.Mv * self.LAMBDA
            for article in self.articles_models:
                denominator += article.Wti[c_idx] * article.Mnt
            print("denominator: " + str(denominator))

            nominator = self.LAMBDA
            for k in range(self.Mv):
                word_k = self.frequent_words[k]
                for article in self.articles_models:
                    wti = article.Wti[c_idx]
                    nominator += wti * article.words_appearances_dict.setdefault(word_k,0)
                self.Piks[c_idx, k] = float(nominator) / float(denominator)
        print(str(self.Piks))
        print("done (" + str(time.time() - s) + ")")


    def compute_z(self):
        print("Computing Z... "),
        s = time.time()
        log_alphas = np.log(self.alphas)
        log_Ps = np.log(self.Piks)
        for article in self.articles_models:
            values = [0]*self.num_clusters
            for c_idx in range(self.num_clusters):
                sum_values = 0
                for k in range(self.Mv):
                    sum_values += article.words_appearances_dict.setdefault(self.frequent_words[k],0)*log_Ps[c_idx,k]
                values[c_idx] = log_alphas[c_idx] + sum_values
            article.set_zValues(values)

        print("done (" + str(time.time() - s) + ")")

    def cluster(self):
        print("clustering... "),
        s = time.time()

        # get initial log likelihood:
        l_before = self.log_likelihood()

        while True:
            # E stage
            self.compute_W()

            # M Stage
            self.compute_alphas()
            self.compute_Piks()
            self.compute_z()

            l_after = self.log_likelihood()

            # sanity check
            if l_after < l_before:
                print("Error! " + str(l_after) + " < " + str(l_before))
                return

            # in case change is too small, stop running
            if l_after - l_before < self.stop_threshold:
                print("Finished running. l_after: " + str(l_after) + " l_before: " + str(l_before)),
                break
            else:
                print(" l_after: " + str(l_after) + " l_before: " + str(l_before)),

            l_before = l_after
        print(" done(" + str(time.time() - s) + ")")

    def log_likelihood(self):
        log_likelihood = 0.0
        for article in self.articles_models:
            z_values = article.get_zValues()
            m = max(z_values)
            sum_values = 0.0
            for i in range(self.num_clusters):
                if z_values[i] - m >= -self.k_value:
                    sum_values += math.exp(z_values[i]-m)
            log_likelihood += m + math.log(sum_values)
        return log_likelihood

    def create_clusters(self):
        clusters = {cluster: [] for cluster in range(self.num_clusters)}
        for article in self.articles_models:
            i = article.get_max_cluster()
            clusters[i].append(article)
        return clusters

    def create_confusion_matrix(self, topics):
        clusters = self.create_clusters()
        sorted_clusters = sorted(clusters.items(), key=lambda e: len(e[1]), reverse=True)

        lines = ["\t".join(topics.values()) + "\tTotal\n"]

        for cluster, articles in sorted_clusters:
            topics_counter = {i: 0 for i in range(self.num_clusters)}
            for article in articles:
                real_topics = article.get_real_topics()
                for topic in real_topics:
                    topics_counter[list(topics.values()).index(topic)] += 1
            print(topics_counter)
            lines.append("\t".join([ str(x) for x in topics_counter.values()]) + "\t" + str(len(articles))+"\n")

        matrix_filename = "matrix.txt"
        matrix_file = open(matrix_filename, 'w')
        matrix_file.writelines(lines)
