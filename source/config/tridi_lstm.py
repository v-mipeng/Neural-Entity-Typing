from multi_time_lstm import BasicConfig
from model.tridir_lstm import TDLM
from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os

class TDLC(BasicConfig):
    Model = TDLM

    model_path = os.path.join(BasicConfig.basedir,"output/models/tridi_lstm.pkl");

    word2id_path = os.path.join(BasicConfig.basedir, "input/tables/word2id.txt")

    word_freq_path = os.path.join(BasicConfig.basedir, "input/tables/word freq.txt")

    embed_path = os.path.join(BasicConfig.basedir, "input/tables/word embedding.txt")

    train_path = os.path.join(BasicConfig.data_path, "test/")

    sparse_word_threshold = 10

    sparse_mention_threshold = 30

    step_rule = AdaDelta(decay_rate = 0.95, epsilon = 1e-06)
