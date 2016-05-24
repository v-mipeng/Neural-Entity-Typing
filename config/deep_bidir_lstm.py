from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Tanh
import math
import os

from model.multi_time_lstm import Model

basedir = r"./";

data_path = os.path.join(basedir,"input");

model_path = os.path.join(basedir,"output/models");

word2id_path = os.path.join(basedir, "input/tables/satori and bbn/word2id.txt")

word_freq_path = os.path.join(basedir, "input/tables/satori and bbn/word freq.txt")

embed_path = os.path.join(basedir, "input/tables/word embedding.txt")

batch_size = 32
sort_batch_count = 20

embed_size = 300

lstm_time = 2
lstm_size = 256

n_labels = 5

step_rule = AdaDelta(decay_rate = 0.95, epsilon = 1e-06)

dropout = 0.0
w_noise = 0.00

valid_freq = 1000
save_freq = 1000
print_freq = 100    # measured by batches

to_label_id = {
u"music.music": 0,
u"broadcast.content": 0,
u"book.written_work": 0,
u"award.award": 0,
u"body.part": 0,
u"chemicstry.chemicstry": 0,
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
