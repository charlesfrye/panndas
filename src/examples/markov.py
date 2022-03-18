import numpy as np
import pandas as pd

from .math import entropy
from .nn import Module
from ..data import text as td


def make_second_order_featurizer(vocab):
    """Create a weight matrix that picks out singlets and pairs from bag-of-words vectors."""

    feature_weights = td.make_second_order_df(vocab)
    indexed_vocab = pd.Series(
        feature_weights.columns, index=td.to_index(vocab)
    )  # vocab as entries and index
    for feature in feature_weights.index:  # feature: single word or a pair of words
        # first, put a 1 wherever any word from the feature appears, 0 elsewhere
        feature_weights.loc[feature] = indexed_vocab.map(
            lambda s: s in feature.split(" ")
        )

        # then, rescale so that single word detectors have value 2
        # and each element in pair has value 1
        feature_weights.loc[feature] *= 4 - len(feature.split(" "))
        feature_weights.loc[feature] -= 1  # (and set all 0 entries to -1)

    return feature_weights


class Transitions:
    def __init__(self, vocab):
        """A transition matrix for a second-order Markov model with skips.

        The maximum skip size is set by window."""
        self.vocab = vocab
        self._data = td.make_second_order_df(vocab).fillna(0)

    def update(self, corpus, window):
        for sentence in corpus:
            self.update_from_sentence(sentence, window)

    def update_from_sentence(self, sentence, window):
        words = sentence.split(" ")
        for skip in range(0, window):
            pairs = td.pairs_from_sentence(sentence, skip=skip)

            for (pair, next) in zip(pairs, words[skip + 1 :]):
                is_singlet = pair[0] == pair[1]  # skip=0
                if is_singlet:  # skip=0
                    feature = pair[0]
                else:
                    feature = " ".join(pair)

                try:
                    self._data.loc[feature, next] += 1
                except KeyError:
                    if not is_singlet:
                        reversed_feature = " ".join(reversed(pair))
                        self._data.loc[reversed_feature, next] += 1

    @classmethod
    def from_corpus(cls, corpus, window=8):
        """Creates Transitions prepopulated with data from a corpus."""
        vocab = td.vocab_from_corpus(corpus)
        t = Transitions(vocab)

        t.update(corpus, window)

        return t

    def to_probabilities_df(self):
        pdf = self._data.apply(normalize_series, axis=1).T
        pdf = pdf.fillna(1 / len(pdf))
        return pdf


def normalize_series(s):
    return s / sum(abs(s))


class SkipWindowModel(Module):
    def __init__(self, transition_matrix, window=8):
        """Optimal model for predicting a second-order Markov model with skips.

        The maximum skip size is set by window."""
        self.transitions = transition_matrix
        self.window = 8

    def forward(self, xs):
        words = xs.split(" ")
        pairs = words[-1:] + td.skips_to_target(words, self.window)

        p = self.get_min_entropy(pairs)

        return p

    def predict(self, xs):
        p = self(xs)

        return p.sample(weights=p).index[0]

    def prompt(self, xs):
        print(xs)
        last = xs.split(" ")[-1]
        stop = "please"
        while not last == stop:
            next = self.predict(xs)
            print(next)
            xs = xs + " " + next
            last = next

    def get_min_entropy(self, pairs):
        best_conditional_entropy, p = np.inf, None

        for possible_pair in self.transitions.columns:
            l_possible_pair = possible_pair.split(" ")
            rev_l_possible_pair = list(reversed(l_possible_pair))
            if (l_possible_pair in pairs) or (rev_l_possible_pair in pairs):

                conditional_p = self.transitions[possible_pair]
                conditional_entropy = entropy(conditional_p)

                if conditional_entropy < best_conditional_entropy:
                    best_conditional_entropy = conditional_entropy
                    p = conditional_p

                if best_conditional_entropy == 0.0:
                    break

        return p
