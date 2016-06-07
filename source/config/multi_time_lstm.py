from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale
import os

from model.multi_time_lstm import MTLM, MTLDM

class BasicConfig:
    debug = False

    develop = True

    basedir = r"./";

    nltk_data_path = os.path.join(basedir, "input/nltk/")

    data_path = os.path.join(basedir,"input");

    model_path = os.path.join(basedir,"output/models/multi_time_lstm_test.pkl");

    word2id_path = os.path.join(basedir, "input/tables/word2id.txt")

    word_freq_path = os.path.join(basedir, "input/tables/word freq.txt")

    embed_path = os.path.join(basedir, "input/tables/word embedding.txt")

    train_path = os.path.join(data_path, "test/")

    valid_portion = 0.05

    test_path = os.path.join(data_path, "test/")

    test_result_path = "./output/result/test/"

    predict_path = os.path.join(data_path, "predict/")

    predict_result_path = "./output/result/predict/"

    int_type = "int32"

    sparse_word_threshold = 10

    sparse_mention_threshold = 50

    batch_size = 32
    sort_batch_count = 20

    embed_size = 300

    lstm_time = 2
    lstm_size = 256

    type_embed_size = 100

    n_labels = 5

    step_rule = AdaDelta(decay_rate = 0.95, epsilon = 1e-06)
    step_rule = CompositeRule([Scale(1.0), BasicMomentum(momentum=0.9)])
    valid_freq = 1000
    save_freq = 1000
    print_freq = 100    # measured by batches

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
    Model = MTLM

    with_pre_train = False

    emb_backup_path = os.path.join(basedir, "input/tables/word embedding backup.txt")

 
class MTLDC(BasicConfig):
    Model = MTLDM

    develop = True

    type2id_path = os.path.join(BasicConfig.basedir, "input/tables/type2id.txt")

    type_embed_size = 100
