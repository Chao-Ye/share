import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import transtab
from dataset import load_data

import warnings
warnings.filterwarnings("ignore")

transtab.random_seed(42)

cal_device = 'cuda:0'
cpt = './checkpoint-scratch'

# use the given load_data
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = load_data('credit-g')

model = transtab.build_classifier(
    # hidden_dropout_prob=0.5,
    device=cal_device
    )

training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    # 'eval_metric':'val_loss',
    # 'eval_less_is_better':True,
    'eval_metric':'auc',
    'eval_less_is_better':False,
    'output_dir':cpt,
    'patience':5,
    'flag':1
    }
transtab.train(model, trainset, valset, **training_arguments)

x_test, y_test = testset
ypred = transtab.predict(model, x_test)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, ypred))
# transtab.evaluate(ypred, y_test, metric='auc')

pdb.set_trace()
pass




