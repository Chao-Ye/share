import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import transtab
from dataset import load_data

import warnings
warnings.filterwarnings("ignore")

transtab.random_seed(42)

cal_device = 'cuda:0'
cpt = './checkpoint-pretrain'
cpt_output = './checkpoint-finetune'

# use the given load_data
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = load_data('credit-g')

model = transtab.build_classifier(
    # hidden_dropout_prob=0.1,
    checkpoint=cpt, # load pretrained model
    device=cal_device
    )

model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4, # 5e-4
    # 'eval_metric':'val_loss',
    # 'eval_less_is_better':True,
    'eval_metric':'auc',
    'eval_less_is_better':False,
    'output_dir':cpt_output,
    'patience':5,
    'flag':1
    }
transtab.train(model, trainset, valset, **training_arguments)

x_test, y_test = testset
ypred = transtab.predict(model, x_test)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, ypred))
# transtab.evaluate(ypred, y_test, metric='auc')