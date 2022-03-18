from itertools import combinations

import pandas as pd

from ..nn.modules import Module


class Tokenizer(Module):
    """Tokenizes by spaces and converts to one-hot vectors."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_ix = pd.Index(vocab)

    def tokenize_one(self, word):
        return pd.Series(self.vocab_ix == word, index=self.vocab_ix)

    def tokenize_many(self, words):
        return pd.DataFrame({word: self.tokenize_one(word) for word in words})

    def tokenize(self, text):
        words = text.split(" ")
        return self.tokenize_many(words)

    def forward(self, xs):
        return self.tokenize(xs)

    @classmethod
    def from_corpus(cls, corpus):
        """Create a Tokenizer by reading a corpus to determine the vocabulary."""
        vocab = vocab_from_corpus(corpus)
        return cls(vocab)


def vocab_from_corpus(corpus):
    """Create a word list from an iterable of texts, where words are split by spaces."""
    return list(set(sum([text.split(" ") for text in corpus], [])))


def text_to_word(text, word):
    """Chop off all parts of a text after the given word."""
    words = text.split(" ")
    index = words.index(word)
    return " ".join(words[: index + 1])


def to_index(vocab):
    """Create a pandas Index out of an iterable of words."""
    return pd.Index(vocab, name="vocab")


def pairs_from(vocab):
    """Create a list of all possible (order-invariant) pairs of words from a vocabulary."""
    return list(combinations(vocab, 2))


def make_pairs_index(vocab):
    """Create a pandas Index for all pairs of words from a vocabulary. See pairs_from."""
    pairs = pairs_from(vocab)
    pairs_ix = pd.Index(f"{fst} {snd}" for (fst, snd) in pairs)
    return pairs_ix


def make_second_order_index(vocab):
    """Create a pandas Index for all pairs and singles of words from a vocabulary."""
    pair_ix = make_pairs_index(vocab)
    second_order_ix = to_index(vocab).append(pair_ix)
    second_order_ix.name = "features"

    return second_order_ix


def pairs_from_text(text, skip=1):
    """Return all pairs of words that occur in a text."""
    words = text.split(" ")
    skip_pairs = zip(words, words[skip:])

    return skip_pairs


def make_second_order_df(vocab):
    """Create an empty DataFrame with vocab in columns and singlets/pairs in rows."""
    vocab_ix = to_index(vocab)
    second_order_ix = make_second_order_index(vocab)
    indexed_vocab = pd.Series(
        vocab, index=vocab_ix, name="vocab"
    )  # vocab as entries and index

    df = pd.DataFrame(index=second_order_ix, columns=indexed_vocab)
    return df


def skips_to_target(words, window):
    """Create the list of paired words before the target within the given window."""
    words, target = words[-window:], words[-1]
    pairs = [[word, target] for word in words[:-1]]

    return pairs
