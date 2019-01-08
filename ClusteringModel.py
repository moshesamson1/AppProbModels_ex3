import numpy as np
import math
import time


class ClusteringModel():
    def __init__(self, articles_models, frequent_words):
        self.eps = 0.0001
        self.stop_threshold = 1
        self.num_clusters = 9
        self.articles_models = articles_models
        self.num_articles = len(self.articles_models)
        self.frequent_words = frequent_words
        self.Mv = len(frequent_words)
        self.indices = []
        self.alphas = []
        self.Piks = np.zeros((self.num_clusters, self.Mv))
        self.LAMBDA = 0.5
        self.k_value = 10


    def init_parameters(self):
        for article_idx in xrange(self.num_articles):
            self.articles_models[article_idx].set_topic_idx((article_idx % self.num_clusters)+1)
            self.articles_models[article_idx].create_ntk(self.frequent_words)

        self.compute_alphas()
        self.compute_Piks()
        self.compute_z()


    def compute_alphas(self):
        print "Computing Alphas... ",
        s = time.time()
        self.alphas = [(sum([article.get_Wti(i) for article in self.articles_models]))/float(self.num_articles) for i in xrange(self.num_clusters)]
        self.alphas = [self.eps if x == 0 else x for x in self.alphas]
        su = sum(self.alphas)
        self.alphas = [x/su for x in self.alphas]
        print "done (" + str(time.time() - s) + ")"


    def compute_Piks(self):
        """
        The probability of a word Wk in topic Ti
        :return:
        """
        print "Computing Pik... ",
        s = time.time()
        for c_idx in xrange(self.num_clusters):
            print "cluster: " + str(c_idx),

            compute_nominator = True
            nominator = self.LAMBDA
            denominator = self.Mv * self.LAMBDA
            for k in xrange(self.Mv):
                for article in self.articles_models:
                    if compute_nominator:
                        nominator += article.get_Wti(c_idx) * article.get_Ntk(self.frequent_words[k])
                    denominator += article.get_Wti(c_idx) * article.get_Mnt()
                self.Piks[c_idx, k] = nominator / denominator
                compute_nominator = False

            # denominator = self.Mv * self.LAMBDA + sum([article.get_Wti(c_idx) * article.get_Mnt() for article in self.articles_models])
            # nominator = self.LAMBDA
            # for k in xrange(self.Mv):
            #     nominator += sum([article.get_Wti(c_idx) * article.get_Ntk(self.frequent_words[k]) for article in self.articles_models])
            #     self.Piks[c_idx, k] = nominator / denominator

        print "done (" + str(time.time() - s) + ")"


    def compute_z(self):
        print "Computing Z... ",
        s = time.time()
        log_alphas = np.log(self.alphas)
        log_Ps = np.log(self.Piks)
        for article in self.articles_models:
            a_Ntk = article.get_Ntk_full()
            article.set_zValues([log_alphas[c_idx] + \
                                           sum([a_Ntk[self.frequent_words[k]]*
                                                log_Ps[c_idx, k] for k in xrange(self.Mv)]) for c_idx in xrange(self.num_clusters)])
        print "done (" + str(time.time() - s) + ")"

    def cluster(self):
        print "clustering... ",
        s = time.time()

        # get initial log likelihood:
        l_before = self.log_likelihood()

        while(True):
            # E stage
            self.compute_W()

            # M Stage
            self.compute_alphas()
            self.compute_Piks()
            self.compute_z()


            l_after = self.log_likelihood()

            #sanity check
            if l_after < l_before:
                print("Error! " + str(l_after) + " < " + str(l_before))
                return

            # in case change is too small, stop running
            if l_after - l_before < self.stop_threshold:
                print("Finished running")
                break


            l_before = l_after
        print(" done(" + str(time.time() - s) + ")")


    def log_likelihood(self):
        log_likelihood = 0.0
        for article in self.articles_models:
            z_values = article.get_zValues()
            m = max(z_values)
            log_likelihood += math.log(sum([math.exp(z_values[i]-m) for i in xrange(self.num_clusters) if z_values[i]-m >= -self.k_value]))
        return log_likelihood

    def compute_W(self):
        print("Computing W... "),
        s = time.time()
        for article in self.articles_models:
            z_values = article.get_zValues()
            m = max(z_values)
            article.set_Wti(
                [0 if z_values[i] - m < self.k_value
                 else
                 (np.exp(z_values[i]-m) /
                  sum([math.exp(z_values[j]-m) for j in xrange(self.num_clusters) if z_values[j]-m >= -self.k_value]))
                 for i in xrange(self.num_clusters)]
            )
        print("done (" + str(time.time() - s) + ")")