from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os
from model.step_rule import WAdaDelta

from model.lstm_model import MTLM, MTLDM, WLSTMM, BDLSTMM, BDLSTMM2

class BasicConfig:
    '''
    Basic Config
    '''
    debug = True

    # True if data is pre-processed, otherwise False
    develop = True              

    basedir = r"./";

    nltk_data_path = os.path.join(basedir, "input/nltk/")

    data_path = os.path.join(basedir,"input");

    model_path = os.path.join(basedir,"output/models/multi_time_lstm_debug.pkl")

    word2id_path = os.path.join(basedir, "input/tables/debug/word2id.txt")

    word_freq_path = os.path.join(basedir, "input/tables/debug/word freq.txt")

    # True: use vectors in embed_path otherwise embed_backup_path to initialize word embeddings
    with_pre_train = True

    embed_path = os.path.join(basedir, "input/tables/word embedding.txt")

    # Small size randomly selected pre-trained embedding
    embed_backup_path = os.path.join(basedir, "input/tables/word embedding backup.txt") 

    # If is directory, read all the files with extension ".txt"
    train_path = os.path.join(data_path, "test/")

    valid_portion = 0.05

    test_path = os.path.join(data_path, "test/")

    test_result_path = "./output/result/test/"

    predict_path = os.path.join(data_path, "predict/")

    predict_result_path = "./output/result/predict/"

    # GPU: "int32"; CPU: "int64"
    int_type = "int32"

    # Do stemming (special)if frequency of word in context except for that in mention < sparse_word_threshold
    sparse_word_threshold = 10

    # Do stemming (special) if frequency of word in mention < sparse_mention_threshold
    sparse_mention_threshold = 10

    batch_size = 32
    sort_batch_count = 20

    embed_size = 300

    lstm_size = 256

    n_labels = 5

    step_rule = AdaDelta(decay_rate = 0.95, epsilon = 1e-06)

    # Measured by batches, e.g, valid every 1000 batches
    valid_freq = 1000
    save_freq = 1000
    print_freq = 100    

    to_label_id = {
    u"music.music": 0,
    u"broadcast.content": 0,
    u"book.written_work": 0,
    u"award.award": 0,
    u"body.part": 0,
    u"chemistry.chemistry": 0,
    u"time.event": 0,
    u"food.food": 0,
    u"language.language": 0,
    u"location.location": 1,
    u"organization.organization": 2,
    u"people.person": 3,
    u"computer.software": 4,
    u"commerce.consumer_product": 4,
    u"commerce.electronics_product": 4,
    }

class MTLC(BasicConfig):
    '''
    Multiple Time LSTM Config
    '''
    Model = MTLM

    with_pre_train = True

    lstm_time = 2
 
class MTLDC(BasicConfig):
    '''
    Multiple Time LSTM with DBpedia Config
    '''
    Model = MTLDM

    develop = True

    type2id_path = os.path.join(BasicConfig.basedir, "input/tables/type2id.txt")

    type_embed_size = 100

class WLSTMC(BasicConfig):
    '''
    Weighted (with Gaussian distribution) Single LSTM Config
    '''
    Model = WLSTMM

    # Define step rule for sigma of gaussian distribution
    step_rule = WAdaDelta(special_para_names = "delta")

    model_path = os.path.join(BasicConfig.basedir,"output/models/weight_lstm.pkl");

class BDLSTMC(BasicConfig):
    '''
    Bi-direction LSTM Config: order_lstm(mention_end)||reverse_lstm(mention_begin)
    '''
    Model = BDLSTMM

    model_path = os.path.join(BasicConfig.basedir,"output/models/bidir_lstm.pkl");

class BDLSTMC2(BasicConfig):
    '''
    Bi-direction LSTM Config: order_lstm(mention_begin-1)||max_pooling(mention)||reverse_lstm(mention_end+1)
    '''
    Model = BDLSTMM2

    model_path = os.path.join(BasicConfig.basedir,"output/models/bidir_lstm2.pkl");