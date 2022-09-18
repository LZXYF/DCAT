import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import DCAT_BERT_GAT, DCAT_BERT_GAT

parser = argparse.ArgumentParser()

parser.add_argument('--isfusion', type=bool, default=True)
parser.add_argument('--logit1', type=bool, default=True)
parser.add_argument('--modeltest', type=str, default=None)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--nb_epochs', type=int, default=120)
parser.add_argument('--bert_init', type=str, default='roberta-base', choices=['roberta-base', 'bert-base-uncased'])
parser.add_argument('--dataset', default='ohsumed', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
parser.add_argument('--checkpoint_dir', default=None)
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--bert_lr', type=float, default=2e-5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--cnn_lr', type=float, default=1e-3)
parser.add_argument('--mil', type=list, default=[15, 50, 90])

ga = None
m2 = None

args = parser.parse_args()

isfusion=args.isfusion
logit1=args.logit1
modeltest=args.modeltest
cnnlr = args.cnn_lr
mil = args.mil
max_length = args.max_length
batch_size = args.batch_size
m = None
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
# gpu = th.device('cuda:0')
gpu = th.device('cpu')

logger.info('Training Arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

if gcn_model == 'gcn':
    model = DCAT_BERT_GAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout, m2=m2, batchsize=batch_size,
                    isfusion=isfusion, logit1=logit1, modeltest=modeltest)
else:
    model = DCAT_BERT_GAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout, m2=m2, batchsize=batch_size)


if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

doc_mask  = train_mask + val_mask + test_mask

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')

g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['global_feature'] = th.zeros((nb_node, model.feat_dim))

logger.info('Build Graph:')
logger.info(str(g))

train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Training
def update_feature():
    global model, g, doc_mask
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['global_feature'][doc_mask] = cls_feat
    return g

optimizer = th.optim.Adam([
        {'params': model.ln.parameters(), 'lr': bert_lr, "weight_decay_rate": 0.00005},
        {'params': model.dense.parameters(), 'lr': bert_lr, "weight_decay_rate": 0.00005},
        {'params': model.linear.parameters(), 'lr': bert_lr, "weight_decay_rate": 0.00005},
        {'params': model.cnnln.parameters(), 'lr': bert_lr, "weight_decay_rate": 0.00005},
        {'params': model.bert_model.parameters(), 'lr': bert_lr,"weight_decay_rate": 0.00005},
        {'params': model.classifier.parameters(), 'lr': bert_lr, "weight_decay_rate": 0.00005},
        {'params': model.textcnn.parameters(), 'lr': cnnlr, "weight_decay_rate": 0.00005},
        {'params': model.gcn.parameters(), 'lr': gcn_lr, "weight_decay_rate": 0.00005}
    ], lr=1e-4
)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=mil, gamma=ga)

def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]

    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['global_feature'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc

# 创建训练器实例
trainer = Engine(train_step)

@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    # 1 调整学习率
    scheduler.step()
    # 2 更新特征状态
    update_feature()
    # 3 清空gpu缓存
    th.cuda.empty_cache()

def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true

# 创建测试集实例
evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # 对训练集进行测试
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    # 对验证集进行测试
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    # 对测试集进行测试
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.3f} loss: {:.3f}  Val acc: {:.3f} loss: {:.3f}  Test acc: {:.3f} loss: {:.3f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if test_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'cnn': model.textcnn.state_dict(),
                'dense' : model.dense.state_dict(),
                'linear': model.linear.state_dict(),
                'ln' : model.ln.state_dict(),
                'cnnln' : model.cnnln.state_dict(),
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = test_acc

log_training_results.best_val_acc = 0
g = update_feature()

# run train
trainer.run(idx_loader_train, max_epochs=nb_epochs)
