from easydl import *
import torch.nn.functional as F
import numpy as np

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out) # why reverse layer?: do sigmoid()-1
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_source_share_weight_onlyentropy( before_softmax, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    # print (after_softmax)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-5), dim=1, keepdim=True)
    entropy_norm = entropy / (np.log(after_softmax.size(1)) )
    # print (entropy_norm)
    weight = entropy_norm
    weight = weight.detach()
    return weight

def hellinger_distance(p, q):
    return  torch.norm((torch.sqrt(p) - torch.sqrt(q)), p=2, dim=1) / np.sqrt(2)




def get_commonness_weight(ps_s, pt_s, ps_t, pt_t, class_temperature=10.0):

    ps_s = F.softmax(ps_s / class_temperature)
    pt_s = F.softmax(pt_s / class_temperature)
    ps_t = F.softmax(ps_t)
    pt_t = F.softmax(pt_t)

    ws = hellinger_distance(ps_s, pt_s).detach()
    wt = hellinger_distance(ps_t, pt_t).detach()

    return ws, wt



def get_entropy(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = nn.Softmax(-1)(before_softmax)
    domain_logit = reverse_sigmoid(domain_out)  # why reverse layer?: do sigmoid()-1
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val +1e-5)
    x = x / (torch.mean(x)+1e-5) # why do this?
    return x.detach()


def normalize_weight01(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def normalize_weight_11(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x*2 - 1
    return x.detach()

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def tensor_l2normalization(q):
    qn = torch.norm(q, p=2, dim=1).detach().unsqueeze(1)
    q = q.div(qn.expand_as(q))
    return q

class C_AccuracyCounter():
    """
    in supervised learning, we often want to count the test accuracy of every class.
    # this is modified for visda
    but the dataset size maybe is not dividable by batch size, causing a remainder fraction which is annoying.
    also, sometimes we want to keep trace with accuracy in each mini-batch(like in train mode)
    this class is a simple class for counting accuracy.

    usage::

        counter = AccuracyCounter()
        iterate over test set:
            counter.addOneBatch(predict, label) -> return accuracy in this mini-batch
        counter.reportAccuracy() -> return accuracy over whole test set
    """
    
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        self.Ccorrect = np.zeros( (1,12) )
        self.Ctotal = np.zeros( (1,12) )

    def addOneBatch(self, predict, label):
        assert predict.shape == label.shape
        b_array = np.argmax(label,1)
        p_array = np.argmax(predict,1)
        index =0
        for element in b_array:
            self.Ctotal[0][element] += 1
            if(b_array[index] == p_array[index]):
                self.Ccorrect[0][element] +=1
            index +=1
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal,self.Ccorrect/self.Ctotal
    
    def reportAccuracy(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float),self.Ccorrect/self.Ctotal