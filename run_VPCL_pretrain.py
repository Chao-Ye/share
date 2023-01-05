import transtab

import warnings
warnings.filterwarnings("ignore")

# set random seed
transtab.random_seed(42)

###############   choice dataset and device   ###################
pretrain_dataset = [
                # 'credit-g', 
                'credit-approval', 
                'dresses-sales', 
                'adult', 
                'cylinder-bands', 
                'telco-customer-churn',
                'data/IO',
                'data/IC',
            ]
task_dataset = [
                'credit-g', 
                # 'credit-approval', 
                # 'dresses-sales', 
                # 'adult', 
                # 'cylinder-bands', 
                # 'telco-customer-churn',
                # 'data/IO',
                # 'data/IC',
            ]
cal_device = 'cuda:0'
cpt = './checkpoint'
cpt1 = './checkpoint1'
################    pretrain    ################
# 1.注意多个数据集预训练时，他们的列名会被集合到一起
# 2.如果某个相同的列名在不同的数据集中被被划分为不同类型（cat/num）,则会造成冲突
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data(pretrain_dataset)

model, collate_fn = transtab.build_contrastive_learner(
    cat_cols, num_cols, bin_cols, 
    supervised=True, # if take supervised CL
    num_partition=2, # num of column partitions for pos/neg sampling
    overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    device=cal_device,
    # hidden_dropout_prob=0.5,
)
training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':cpt,
    'patience':5,
    'num_workers':4
    }
transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

#################    finetuning and eval    #################
######## finetuning
allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data(task_dataset)

model = transtab.build_classifier(
    checkpoint=cpt,
    # hidden_dropout_prob=0.5,
    device=cal_device
    )
model.update({'cat':cat_cols, 'num':num_cols, 'bin':bin_cols})
training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    # 'eval_metric':'val_loss',
    # 'eval_less_is_better':True,
    'eval_metric':'auc',
    'eval_less_is_better':False,
    'output_dir':cpt1,
    'patience':5,
    'flag':1
    }
transtab.train(model, trainset, valset, **training_arguments)

####### eval
# 注意获得的testset是一个list
print('###################### test data result ######################')
x_test, y_test = testset[0]
ypred = transtab.predict(model, x_test)
transtab.evaluate(ypred, y_test, metric='auc')
print('###################### test data result ######################')