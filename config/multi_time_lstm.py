from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Tanh
import math
import os

from model.multi_time_lstm import Model

basedir = r"D:\Codes\Project\EntityTyping\Neural Entity Typing";

data_path = os.path.join(basedir,"input");

model_path = os.path.join(basedir,"output/models");

embed_path = r"D:\Data\Google-word2vec\GoogleNews-vectors-negative300-selected.txt"

batch_size = 32
sort_batch_count = 20

embed_size = 300

lstm_time = 2
lstm_size = 256

n_labels = 15

step_rule = AdaDelta(decay_rate=0.95, epsilon=1e-06)
dropout = 0.0
w_noise = 0.00

valid_freq = 1000
save_freq = 1000
print_freq = 100    # measured by batches

to_label_id = {
u"music.music": 0,
u"broadcast.content": 1,
u"book.written_work": 2,
u"award.award": 3,
u"body.part": 4,
u"chemicstry.chemicstry": 5,
u"time.event": 6,
u"food.food": 7,
u"language.language": 8,
u"location.location": 9,
u"organization.organization": 10,
u"people.person": 11,
u"computer.software": 12,
u"commerce.consumer_product": 13,
u"commerce.electronics_product": 14,
}
