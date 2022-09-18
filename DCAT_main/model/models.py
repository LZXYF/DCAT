import torch as th
from torch import nn
import torchtext.vocab as vocab
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModel, BertForSequenceClassification, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT
import pickle


global_outsize = 50

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        global_feature = self.bert_model(input_ids, attention_mask)[0][:, 0]
        global_feature = th.nn.Dropout(0.64)(global_feature)
        cls_logit = self.classifier(global_feature)
        return cls_logit

class MHSA(nn.Module):
  def __init__(self, num_heads, dim):
    super().__init__()
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.num_heads = num_heads

  def forward(self, x):
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1)

    v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
    return v


class textCNNModel(nn.Module):

    def __init__(self, pretrained_model, kernel, num_classes, batch_size=64, out_size=100, use_bert=False, input_size=0):
        super().__init__()

        embedding_dict = None
        self.use_bert = use_bert

        with open("./embdict/" + pretrained_model + "_embedding_dict.pkl", 'rb') as ff:
            embedding_dict = pickle.load(ff)

        embed_dim = 0
        if use_bert:
            embed_dim=input_size  # 150
        else:
            embed_dim=200
        kernel_num = kernel  # 50
        Ci = 1
        kernel_sizes = [2, 4]
        self.class_num = num_classes
        class_num = num_classes

        self.embedding = th.nn.Embedding.from_pretrained(embedding_dict)
        self.embedding.weight.requires_grad = True
        

        self.convs_list = th.nn.ModuleList(
            [th.nn.Conv2d(Ci, kernel_num, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])

        atten_size = global_outsize
        self.dropout = th.nn.Dropout(0.6)
        self.bn = th.nn.BatchNorm1d(len(kernel_sizes) * kernel_num, affine=True, eps=1e-07, momentum=0.01)
        self.ln = th.nn.LayerNorm(len(kernel_sizes) * kernel_num)
        self.fc = th.nn.Linear(len(kernel_sizes) * kernel_num, 768)
        self.fc2 = th.nn.Linear(len(kernel_sizes) * kernel_num, atten_size)
        self.atten_fc = th.nn.Parameter(th.randn([batch_size, atten_size, atten_size]), requires_grad=True)
        self.atten_fc2 = th.nn.Parameter(th.randn([batch_size, atten_size, 1]), requires_grad=True)
        self.atten_fc_2 = th.nn.Parameter(th.randn([batch_size, atten_size, atten_size]), requires_grad=True)
        self.atten_fc_2_2 = th.nn.Parameter(th.randn([batch_size, atten_size, 1]), requires_grad=True)
        self.classess = th.nn.Linear(atten_size, class_num)
        self.MHSA = MHSA(10,kernel_num )
        self.init_weight()

    # 初始化权重
    def init_weight(self):
        init_range = 0.2

        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

        self.classess.weight.data.uniform_(-init_range, init_range)
        self.classess.bias.data.zero_()

        init_range = 0.2
        self.MHSA.q.weight.data.uniform_(-init_range, init_range)
        self.MHSA.q.bias.data.zero_()
        self.MHSA.k.weight.data.uniform_(-init_range, init_range)
        self.MHSA.k.bias.data.zero_()
        self.MHSA.v.weight.data.uniform_(-init_range, init_range)

    def forward(self, x, mask):
        if not self.use_bert:
            x = self.embedding(x)
        x = x * mask.unsqueeze(-1)
        x = x.unsqueeze(1)
        x = [F.relu(F.dropout(conv(x), training=self.training, p=0.3)).squeeze(3) for conv in self.convs_list]
        x = [F.relu(F.dropout(F.max_pool1d(i, i.size(2)).permute(0, 2, 1), training=self.training, p=0.2)) for i in x]
        x = th.cat(x, 1)

        if self.training:
            x = F.relu(self.dropout(x))

        o = x.view(x.size(0), -1)

        x = self.MHSA(x)

        logit = x.view(x.size(0), -1)
        logit =  self.ln(logit + o)
        logit = self.fc(logit)
        return logit

    def atten_fusion(self, input_cnn, input_bert, input_gcn):

        input_cnn = th.unsqueeze(input_cnn, dim=0)
        input_bert = th.unsqueeze(input_bert, dim=0)
        input_gcn = th.unsqueeze(input_gcn, dim=0)
        inputs = th.cat([input_cnn, input_bert, input_gcn], dim=0)
        t = inputs.shape[1]
        inputs = inputs.permute(1, 0, 2)
        dkk = []
        outputs = []
        for x in range(t):
            t_x = inputs[x, :, :]
            x_proj = th.matmul(t_x, self.atten_fc[x])
            x_proj = nn.Tanh()(x_proj)
            u_w = self.atten_fc2[x]
            x = th.matmul(x_proj, u_w)
            alphas = nn.Softmax(dim=-1)(x)
            dkk.append(alphas)

            output = th.matmul(t_x.permute(1, 0), alphas)
            output = th.squeeze(output, dim=-1)
            outputs.append(output)

        final_output = th.stack(outputs, dim=0)
        return final_output

    def atten_fusion2(self, input_cnn, input_fu):

        input_cnn = th.unsqueeze(input_cnn, dim=0)
        input_fu = th.unsqueeze(input_fu, dim=0)

        inputs = th.cat([input_cnn, input_fu], dim=0)
        t = inputs.shape[1]
        inputs = inputs.permute(1, 0, 2)

        outputs = []
        for x in range(t):
            t_x = inputs[x, :, :]
            x_proj = th.matmul(t_x, self.atten_fc_2[x])
            x_proj = nn.Tanh()(x_proj)
            u_w = self.atten_fc_2_2[x]
            x = th.matmul(x_proj, u_w)
            alphas = nn.Softmax(dim=0)(x)

            output = th.matmul(t_x.permute(1, 0), alphas)
            output = th.squeeze(output, dim=-1)
            outputs.append(output)

        final_output = th.stack(outputs, dim=0)
        return final_output

    def atten_fusion3(self, input_cnn, input_fu):

        input_cnn = th.unsqueeze(input_cnn, dim=0)
        input_fu = th.unsqueeze(input_fu, dim=0)
        inputs = th.cat([input_cnn, input_fu], dim=0)
        t = inputs.shape[1]

        inputs = inputs.permute(1, 0, 2)
        outputs = []
        for x in range(t):
            t_x = inputs[x, :, :]
            x_proj = th.matmul(t_x, self.atten_fc[x])
            x_proj = nn.Tanh()(x_proj)
            u_w = self.atten_fc2[x]
            x = th.matmul(x_proj, u_w)
            alphas = nn.Softmax(dim=0)(x)
            output = th.matmul(t_x.permute(1, 0), alphas)
            output = th.squeeze(output, dim=-1)
            outputs.append(output)

        final_output = th.stack(outputs, dim=0)
        return final_output

class DCAT_BERT_GCN(th.nn.Module):

    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.6, gcn_layers=2, n_hidden=200, dropout=0.5, m2=0.2, batchsize=64, isfusion=True, logit1=False, modeltest=None):

        super(DCAT_BERT_GCN, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        
        outsize = global_outsize
        base_outsize = outsize 

        self.textcnn = textCNNModel(pretrained_model, 100, nb_class, batch_size=batchsize, out_size=outsize, use_bert=False, input_size=2*50)
        # bert 默认 768
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        
        self.classifier = th.nn.Linear(self.feat_dim, outsize)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=outsize,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

        self.isfusion = isfusion
        self.logit1 = logit1
        self.modeltest=modeltest

        self.bn = th.nn.BatchNorm1d(base_outsize  , affine=True, eps=1e-07, momentum=0.01)
        self.ln = th.nn.LayerNorm(base_outsize)
        self.linear = th.nn.Linear(base_outsize*2, nb_class)
        self.cnnln = th.nn.Linear(768, base_outsize)
        self.dense = th.nn.Linear(base_outsize, nb_class)

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        
        text_logit = self.textcnn(input_ids, attention_mask)
        if self.training:     
            global_feature = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['global_feature'][idx] = global_feature
        else:
            global_feature = g.ndata["global_feature"][idx]

        cls_logit = self.classifier(global_feature)
        feats = g.ndata["global_feature"]
        gcn_logit = self.gcn(feats, g, g.edata['edge_weight'])[idx]
        text_logit = self.cnnln(text_logit)
        
        logit1 = self.textcnn.atten_fusion2(text_logit, gcn_logit)
        logit2 = self.textcnn.atten_fusion3(gcn_logit, cls_logit)

        if self.isfusion:
            logit2 = th.cat([self.ln(logit1), self.ln(logit2)], dim=1)
            finalout = self.linear(logit2)
        elif self.logit1:
            finalout = self.dense(self.ln(logit1))
        else:
            finalout = self.dense(self.ln(logit2))

        if self.modeltest is not None:
            if self.modeltest == "gcn":
                finalout = self.dense(self.ln(gcn_logit))
            elif self.modeltest == "bert":
                finalout = self.dense(self.ln(cls_logit))
            elif self.modeltest == "cnn":
                finalout = self.dense(self.ln(text_logit))

        pred = th.nn.Softmax(dim=1)(finalout)
        pred = th.log(pred)
    
        return pred
    
class DCAT_BERT_GAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, 
        heads=8, n_hidden=32, dropout=0.5, m2=0.2, batchsize=64, isfusion=True, logit1=False, modeltest=None):
        super(DCAT_BERT_GAT, self).__init__()

        outsize = global_outsize
        base_outsize = outsize

        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, outsize)

        self.textcnn = textCNNModel(pretrained_model, 100, nb_class, batch_size=batchsize, out_size=outsize, use_bert=False, input_size=2*50)
        self.classifier = th.nn.Linear(self.feat_dim, outsize)

        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=outsize,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )
        self.isfusion = isfusion   # wather fusion of model
        self.logit1 = logit1   # when isfusion = False only use logit1 (gcn and cnn)
        self.modeltest=modeltest   # test single model

        self.ln = th.nn.LayerNorm(base_outsize)
        self.linear = th.nn.Linear(base_outsize * 2, nb_class)
        
        self.dense = th.nn.Linear(base_outsize, nb_class)

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            global_feature = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['global_feature'][idx] = global_feature
        else:
            global_feature = g.ndata['global_feature'][idx]

        cls_logit = self.classifier(global_feature)
        gcn_logit = self.gcn(g.ndata['global_feature'], g)[idx]
        text_logit = self.textcnn(input_ids, attention_mask)
        
        logit1 = self.textcnn.atten_fusion2(gcn_logit, text_logit)
        logit2 = self.textcnn.atten_fusion3(gcn_logit, cls_logit)

        if self.isfusion:
            logit2 = th.cat([self.ln(logit1), self.ln(logit2)], dim=1)
            finalout = self.linear(logit2)
        elif self.logit1:
            finalout = self.dense(self.ln(logit1))
        else:
            finalout = self.dense(self.ln(logit2))

        if self.modeltest is not None:
            if self.modeltest == "gcn":
                finalout = self.dense(self.ln(gcn_logit))
            elif self.modeltest == "bert":
                finalout = self.dense(self.ln(cls_logit))
            elif self.modeltest == "cnn":
                finalout = self.dense(self.ln(text_logit))

        pred = th.nn.Softmax(dim=1)(finalout)
        pred = th.log(pred)

        return pred
