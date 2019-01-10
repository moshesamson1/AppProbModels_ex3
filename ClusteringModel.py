import numpy as np
import math
import time





class ClusteringModel:
    def __init__(self, articles_models, frequent_words, words_dict):
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
        self.N = 0.0
        self.words_dict = words_dict
        self.likelihood_data = []
        self.perplexity_data = []
        self.likelihood_filename = 'likelihood.txt'
        self.perplexity_filename = 'perplexity.txt'

    def init_parameters(self):
        self.N = sum([self.words_dict[word] for word in self.frequent_words])

        for article_idx in range(self.num_articles):
            self.articles_models[article_idx].set_topic_idx((article_idx % self.num_clusters))
            self.articles_models[article_idx].create_ntk(self.frequent_words)

        self.compute_alphas()
        self.compute_p()
        self.compute_z()

    def compute_w(self):
        print("Computing W... ", end='')
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
        print("Computing Alphas... ", end='')
        s = time.time()
        self.alphas = [(sum([article.Wti[i] for article in self.articles_models]))/float(self.num_articles)
                       for i in range(self.num_clusters)]
        self.alphas = [self.eps if x < self.eps else x for x in self.alphas]
        su = sum(self.alphas)
        self.alphas = [x/su for x in self.alphas]
        print("done (" + str(time.time() - s) + ")")

    def compute_p(self):
        """
        The probability of a word Wk in topic Ti
        :return:
        """
        print("Computing Pik... ", end='')
        s = time.time()
        for c_idx in range(self.num_clusters):
            print(" cluster: " + str(c_idx), end='', flush=True)

            # denominator = self.Mv * self.LAMBDA
            denominator = self.Mv * self.LAMBDA + sum([article.Wti[c_idx] * article.Mnt for article in self.articles_models])
            # for article in self.articles_models:
            #     denominator += article.Wti[c_idx] * article.Mnt

            for k in range(self.Mv):
                nominator = self.LAMBDA
                word_k = self.frequent_words[k]
                for article in self.articles_models:
                    wti = article.Wti[c_idx]
                    nominator += wti * article.words_appearances_dict.setdefault(word_k, 0)
                self.Piks[c_idx, k] = float(nominator) / float(denominator)

        print(" done (" + str(time.time() - s) + ")")

    def compute_z(self):
        print("Computing Z... ", end='')
        s = time.time()
        log_alphas = np.log(self.alphas)
        log_p = np.log(self.Piks)
        for article in self.articles_models:
            values = [0]*self.num_clusters
            for c_idx in range(self.num_clusters):
                sum_values = 0
                for k in range(self.Mv):
                    sum_values += article.words_appearances_dict.setdefault(self.frequent_words[k], 0)*log_p[c_idx, k]
                values[c_idx] = log_alphas[c_idx] + sum_values
            article.set_zValues(values)

        print("done (" + str(time.time() - s) + ")")

    def cluster(self):
        print("clustering... ")
        s = time.time()

        iteration = 1
        # get initial log likelihood:
        l_before = self.log_likelihood()

        while True:
            # E stage
            self.compute_w()

            # M Stage
            self.compute_alphas()
            self.compute_p()
            self.compute_z()

            l_after = self.log_likelihood()
            print("({:.0f}) log_L: {:.2f} -> {:.2f}".format(iteration, l_before, l_after))

            # save values:
            self.save_likelihood_and_preplexity(iteration, l_after)

            # sanity check
            assert l_after >= l_before

            # in case change is too small, stop running
            if l_after - l_before < self.stop_threshold:
                print("Finished clustering."),
                break

            l_before = l_after
            iteration += 1

        self.write_likelihood_and_preplexity()
        print(" done(" + str(time.time() - s) + ")")

    def save_likelihood_and_preplexity(self, iteration, likelihood):
        self.likelihood_data.append(str(iteration) + "," + str(likelihood))
        self.perplexity_data.append(str(iteration) + "," + str(likelihood))

    def write_likelihood_and_preplexity(self):
        # write log likelihood
        fl = open(self.likelihood_filename, 'w')
        fl.writelines(self.likelihood_data)
        fl.close()

        # write perplexity
        fp = open(self.perplexity_filename, 'w')
        fp.writelines(self.perplexity_data)
        fp.close()

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

    def get_perplexity(self, l_likelihood):
        return math.exp((-1.0 / self.N) * l_likelihood)

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

        table = dict()
        for cluster, articles in sorted_clusters:
            topics_counter = {i: 0 for i in range(self.num_clusters)}
            for article in articles:
                real_topics = article.get_real_topics()
                for topic in real_topics:
                    topics_counter[list(topics.values()).index(topic)] += 1
            lines.append("\t".join([str(x) for x in topics_counter.values()]) + "\t" + str(len(articles))+"\n")
            table[cluster] = topics_counter

        matrix_filename = "matrix.txt"
        matrix_file = open(matrix_filename, 'w')
        matrix_file.writelines(lines)

        # perform assignment
        assignments = {}
        for cluster in table.keys():
            am = np.argmax(table[cluster])
            print("cluster " + str(cluster) + " is assigned to article " + topics[am])
            assignments[cluster] = topics[am]

        # calculate accuracy
        accuracy = 0.0
        for cluster, articles in sorted_clusters:
            for article in articles:
                if assignments[cluster] in article.get_real_topics():
                    accuracy += 1

        print("accuracy: {:.2f}".format(accuracy/float(self.num_articles)))

        return table
