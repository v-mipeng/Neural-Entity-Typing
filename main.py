import sys
sys.path.append("./source")
from entrance.lstm_entrance import MTLE, WLSTME, BDLSTME, BDLSTME2

model = BDLSTME()

# Train model with default training dataset
#model.train(train_path = None, valid_portion = None, valid_path = None, model_path = None)

# Test on default dataset
model.test(test_path = None, test_result_path = None, model_path = None)

## Predict on default dataset
#model.predict(predict_path = None, predict_result_path = None, model_path = None)
