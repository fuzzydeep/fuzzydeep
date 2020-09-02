from source.model.base_model import BaseModel
import torch.nn as nn
from copy import deepcopy
import torch
from source.utils.misc import Pack
from collections import OrderedDict
class MPL(BaseModel):
    def __init__(self,
                 model_KB, model_Dialog, model_T
    ):
        super(MPL, self).__init__()

        self.model_KB = model_KB
        self.model_Dialog = model_Dialog
        self.model_T = model_T
        self.max_grad_norm = 2.0
        self.model_num = 2
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.grad_clip = 2.0
    def train_one_batch(self, model, turn_inputs, kb_inputs, train=True, optimizer=None, grad_clip=None):
        metrics_list_S, total_loss = model.iterate(turn_inputs, kb_inputs, optimizer=optimizer,
                                                   grad_clip=grad_clip, is_training=False)

        return total_loss, metrics_list_S

    def learning_phrase_1(self, model, turn_inputs, kb_inputs, optimizer=None, grad_clip=None,
                             mode=None):

        #student
        s_loss, metric_s = self.train_one_batch(self.model_T, turn_inputs, kb_inputs, train=False,
                                                        optimizer=optimizer[mode], grad_clip=grad_clip)
        #teacher
        t_loss, metric_t = self.train_one_batch(model, turn_inputs, kb_inputs, train=False, optimizer=optimizer['T'],
                                                        grad_clip=grad_clip)  # 更新模型参数
        optimizer[mode].zero_grad()
        optimizer['T'].zero_grad()
        train_loss2 = 0
        for i in range(len(metric_t)):
            logits_s = metric_s[i].logits
            prob_t = metric_t[i].prob
            loss_model = model.collect_mpl_metric(inputs = logits_s, target = prob_t)
            train_loss2 += loss_model

        train_loss1 = s_loss
        # train_loss2 = model.collect_metric(metric_s.logits, metric_t.logits)
        #TODO train_loss3

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        train_loss.backward()
        nn.utils.clip_grad_norm_(self.model_T.parameters(), self.max_grad_norm)
        optimizer['T'].step()



    def learning_phrase_2(self,model, turn_inputs, kb_inputs, optimizer=None, grad_clip=None,
                             mode=None):

        lr = 0.0005
        # student
        s_loss, metric_s = self.train_one_batch(self.model_T, turn_inputs, kb_inputs, train=False,
                                                          optimizer=optimizer['T'], grad_clip=grad_clip)
        # teacher
        t_loss, metric_t = self.train_one_batch(model, turn_inputs, kb_inputs, train=False,
                                                          optimizer=optimizer[mode],
                                                          grad_clip=grad_clip)  # 更新模型参数
        optimizer[mode].zero_grad()
        optimizer['T'].zero_grad()

        train_loss1 = s_loss
        train_loss2 = 0
        for i in range(len(metric_t)):
            logits_s = metric_s[i].logits
            prob_t = metric_t[i].prob
            loss_model = model.collect_mpl_metric(inputs=logits_s, target=prob_t)
            train_loss2 += loss_model

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
      
        fast_weights = OrderedDict((name, param) for (name, param) in self.model_T.named_parameters())

        weights_original = deepcopy(self.model_T.state_dict())
       
        grads = torch.autograd.grad(train_loss, self.model_T.parameters(), create_graph=True,allow_unused=True)
        data = [p.data for p in list(self.model_T.parameters())]

        fast_weights = OrderedDict((name, param - lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

      
        model_dict = self.model_T.state_dict()
        state_dict = {k: v for k, v in fast_weights.items() }
        model_dict.update(state_dict)
        self.model_T.load_state_dict(model_dict)
        
        t_loss,  metric_t = self.train_one_batch(self.model_T, turn_inputs, kb_inputs, train=False,
                                                          optimizer=optimizer['T'],
                                                          grad_clip=grad_clip)
        
        train_loss1 = t_loss
        (torch.mean(train_loss1)).backward()
        optimizer[mode].step()

        self.model_T.load_state_dict({name: weights_original[name] for name in weights_original})






