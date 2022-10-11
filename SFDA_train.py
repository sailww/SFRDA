import torch
from data import *
from net import *
from lib import  *
from loss import *
from torch import optim
from APM_update import *
import torch.backends.cudnn as cudnn
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


cudnn.benchmark = True
cudnn.deterministic = True
def calculate_entropyH2(max_entropy):
    if max_entropy >= 0.5:
        entropy_re = - max_entropy * np.log(max_entropy + 1e-10)+ (- 
            (1-max_entropy)*np.log((1-max_entropy)+1e-10))
    else:
         entropy_re = -  (1-max_entropy) * np.log(max_entropy + 1e-10)+ (- 
            max_entropy*np.log((1-max_entropy)+1e-10))
    return entropy_re
#H6
def calculate_sum(list_max):
    i = 0
    sum =0 
    while(i < len(list_max)):
        sum += list_max[i]
        i = i+1
    return sum
def calculate_p2(val):
    return val *val
def calculate_renyi_entropy(list_val):
    list_max = [calculate_p2(x) for _,x in enumerate(list_val)]
    result = - np.log(calculate_sum(list_max))
    return result
def calculate_logit(list_val):
    total = 0
    for ele in range(0, len(list_val)):
        total = total + list_val[ele]
    return total
def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
def calculate_entropy(max_entropy):
    entropy_re = - max_entropy * np.log(max_entropy + 1e-10)
    return entropy_re
def getmax_secondmax(x):
    top_k_value = np.sort(x)
    return top_k_value[-1] , top_k_value[-2]

seed_everything()
# /home/hanzhongyi/wangfan/TrainSourceModelaccuracy/TrainSourceModelaccBEST_model_checkpoint0.pth.tar
save_model_path = '/home/hanzhongyi/wangfan/TrainSourceModelaccuracy/pretrain_weights_office_home/accBEST_model_checkpointhome32.pth.tar'
print("model is <<<<<<<<<<<")
print(save_model_path)
assert os.path.exists(save_model_path), "{} path does not exist.".format(save_model_path)
save_model_statedict = torch.load(save_model_path)['state_dict']

model_dict = {
    'resnet50': ResNet50Fc,
    'resnet101': ResNet101Fc,
}


# ======= network architecture =======
class Source_FixedNet(nn.Module):
    def __init__(self):
        super(Source_FixedNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = args.data.dataset.n_share
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)

class Target_TrainableNet(nn.Module):
    def __init__(self):
        super(Target_TrainableNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = args.data.dataset.n_share
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.cls_multibranch = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


# ======= pre-trained source network =======
#定义model44`
fixed_sourceNet = Source_FixedNet()
#将预定义的weights给model
fixed_sourceNet.load_state_dict(save_model_statedict)
fixed_feature_extractor_s =(fixed_sourceNet.feature_extractor).cuda()
fixed_classifier_s = (fixed_sourceNet.classifier).cuda()
fixed_feature_extractor_s.eval()
fixed_classifier_s.eval()

# ======= trainable target network =======
trainable_tragetNet = Target_TrainableNet()
feature_extractor_t =(trainable_tragetNet.feature_extractor).cuda()
feature_extractor_t.load_state_dict(fixed_sourceNet.feature_extractor.state_dict())
classifier_s2t = (trainable_tragetNet.classifier).cuda()
classifier_s2t.load_state_dict(fixed_sourceNet.classifier.state_dict())
classifier_t = (trainable_tragetNet.cls_multibranch).cuda()
classifier_t.load_state_dict(fixed_sourceNet.classifier.state_dict())


model_dict = {
            'global_step':0,
            'state_dict': trainable_tragetNet.state_dict(),
            'accuracy': 0}


feature_extractor_t.train()
classifier_s2t.train()
classifier_t.train()
print ("Finish model loaded...")
print(os.environ["CUDA_VISIBLE_DEVICES"])


# domains=['amazon', 'dslr', 'webcam']

# print ('domain....'+domains[args.data.dataset.source]+'>>>>>>'+domains[args.data.dataset.target])

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=(args.train.min_step))

optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor_t.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_s2t = OptimWithSheduler(
    optim.SGD(classifier_s2t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_t= OptimWithSheduler(
    optim.SGD(classifier_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler) 

global_step = 0
best_acc = 0
epoch_id = 0
r = 0.9
p = 1.1
class_num =  args.data.dataset.n_total

##加载数据
_,_,target_train_dl,target_test_dl,_,_ = loaddata()
print("data is loading")
#ceshiyib
if(args.data.dataset.name == 'visda'):
    counter = C_AccuracyCounter()
else:
    counter = AccuracyCounter()
with TrainingModeManager([feature_extractor_t, classifier_t], train=False) as mgr, torch.no_grad():

        for i, (img, label) in enumerate(target_test_dl):
            img = img.cuda()
            label = label.cuda()

            feature = feature_extractor_t.forward(img)
            _, _, _, predict_prob_t = classifier_t.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob_t), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))
        if(args.data.dataset.name == 'visda'):
            acc_test,c_test = counter.reportAccuracy()
            print('>>>>>>>>>>>first accuracy>>>>>>>>>>>>>>>>.')
            print(acc_test,c_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
        else:
            acc_test = counter.reportAccuracy()
            print('>>>>>>>>>>>first accuracy>>>>>>>>>>>>>>>>.')
            print(acc_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

    

while epoch_id < 300:
    epoch_id += 1

    print('epochid : {} is processing'.format(epoch_id))
    # if(epoch_id > 1 and p < 1.2):
    #         p = p + 0.005
    #         print(p)
    prototype_memory, num_prototype_,prototype_memory_dict,select_classes_entropy ,select_classes_energy= APM_init_update(feature_extractor_t, classifier_t,p,r,target_train_dl,epoch_id)
    for i, (img_target, label_target) in enumerate(target_train_dl):
        # if(global_step % pt_memory_update_frequncy == 0):
            
        img_target = img_target.cuda()
        # forward pass:  target network
        fc1_t = feature_extractor_t.forward(img_target)
        _, _, logit_s2t, after_softmax_s2t = classifier_s2t.forward(fc1_t)
        _, _, logit_t,after_softmax_t = classifier_t(fc1_t)
        ###
       
        # # 转成array
        after_softmax = after_softmax_t.detach().cpu().numpy()
         #每行的最大值，最值得相信的那个值
        after_softmax = after_softmax.max(axis = 1 )
        list_entropy = [calculate_entropyH2(x) for _,x in enumerate(after_softmax)]
        list_entropy = list_entropy /np.log(after_softmax_t.size(1))


        
        # compute pseudo labels
        proto_feat_tensor = torch.Tensor(prototype_memory) # (B * 2048)
        feature_embed_tensor = fc1_t.cpu()
        proto_feat_tensor = tensor_l2normalization(proto_feat_tensor)
        batch_feat_tensor = tensor_l2normalization(feature_embed_tensor)

        sim_mat = torch.mm(batch_feat_tensor, proto_feat_tensor.permute(1,0))
        sim_mat = F.avg_pool1d(sim_mat.unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0)# (B, #class)
        pseudo_label_t = torch.argmax(sim_mat, dim=1).cuda()
        #获取次小值，作为均值
        diss = 1 - sim_mat
        first_avg_min = torch.min(diss,dim=1)
        first_avg_min =  first_avg_min[0].detach().cpu().numpy().tolist()
        #获取最小值的位置
        min_val = torch.argmin(diss,dim =1)
        ##
        val = torch.from_numpy(np.array(10))
        min_val = min_val.detach().cpu().numpy().tolist()
        in_sec  = []
        for cls in range(len(min_val)):
            in_sec.append(min_val[cls])
        indexs = (torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])),torch.LongTensor(in_sec)
        diss.index_put_((indexs),torch.Tensor([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]))


        #此时再获取最小值，即为次小值
        second_avg_min = torch.min(diss,dim =1)
        #将其转为数组
        second_avg_min = second_avg_min[0].detach().cpu().numpy().tolist()
        


        
        # confidence-based filtering
        arg_idxs = torch.argsort(sim_mat, dim=1, descending=True) # (B, #class)

        first_group_idx = arg_idxs[:, 0]
        second_group_idx = arg_idxs[:, 1]

        first_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in first_group_idx]
        first_group_feat_tensor = torch.tensor(np.concatenate(first_group_feat, axis=0)) # (B*P, 2048)
        first_group_feat_tensor = tensor_l2normalization(first_group_feat_tensor)

        second_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in second_group_idx]
        second_group_feat_tensor = torch.tensor(np.concatenate(second_group_feat, axis=0)) # (B*P, 2048)
        second_group_feat_tensor = tensor_l2normalization(second_group_feat_tensor)

        feature_embed_tensor_repeat = torch.Tensor(np.repeat(feature_embed_tensor.cpu().data.numpy(), repeats=num_prototype_, axis=0))
        feature_embed_tensor_repeat = tensor_l2normalization(feature_embed_tensor_repeat)

        first_dist_mat = 1 - torch.mm(first_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0)) 
        first_similar_mat = 1 - torch.mm(first_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0)) # distance = 1  - simialirty
        second_dist_mat = 1 - torch.mm(second_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0))

        first_dist_mat = F.max_pool2d(first_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)
        first_similar_mat = -1*F.max_pool2d(-1 *first_similar_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)
        second_dist_mat = -1*F.max_pool2d(-1* second_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)

        first_dist_vec = torch.diag(first_dist_mat) #(B)
        first_similar_vec = torch.diag(first_similar_mat)
        second_dist_vec = torch.diag(second_dist_mat) # B
       



        first_dist_vec = first_dist_vec.numpy().tolist()
        first_similar_vec = first_similar_vec.numpy().tolist()
        second_dist_vec = second_dist_vec.numpy().tolist()
        confidence_mask = []
        pseudo_label = pseudo_label_t.tolist()
        
        for cls in range(args.data.dataloader.batch_size):
            #当前所属类的熵
            if(first_avg_min[cls]*(1.2) <= second_avg_min[cls] or list_entropy[cls] <= select_classes_entropy[pseudo_label[cls]] ):
                confidence_mask.append(1)
            else:
                confidence_mask.append(0)
        confidence_mask = torch.Tensor(confidence_mask)
        confidence_mask = confidence_mask.cuda()
        confidence_mask = confidence_mask.float()
    
        # optimize target network using two types of loss
        # 计算IMloss
        entropy_loss = torch.mean(torch.sum(- after_softmax_s2t* torch.log(after_softmax_s2t + 1e-10), dim=1, keepdim=True))
        msoftmax = after_softmax_s2t.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-10))
        entropy_loss -= gentropy_loss
        #计算 cross-entropy loss
        ce_from_t = nn.CrossEntropyLoss(reduction='none')(logit_t, pseudo_label_t).view(-1, 1).squeeze(1)
        # ce_from_t = CrossEntropyLabelSmooth(num_classes=args.data.dataset.n_share, epsilon=0.1)(logit_t, pseudo_label_t).view(-1, 1).squeeze(1) 
        ce_from_t = torch.mean(ce_from_t.float() * confidence_mask, dim=0, keepdim=True)

        # dynamic optimization
        alpha = np.float(2.0 / (1.0 + np.exp(-global_step/ float((args.train.min_step)/3))) - 1.0)
        ce_total = alpha * entropy_loss + (1-alpha) * ce_from_t

        with OptimizerManager([optimizer_finetune, optimizer_classifier_s2t, optimizer_classifier_t]):
            loss = ce_total
            loss.backward()

        global_step += 1

        # evaluation during training
    if (epoch_id ==1 or epoch_id % 3 == 0):
            counter = AccuracyCounter()
            with TrainingModeManager([feature_extractor_t,classifier_t], train=False) as mgr, torch.no_grad():

                for i, (img, label) in enumerate(target_test_dl):
                    img = img.cuda()
                    label = label.cuda()

                    feature = feature_extractor_t.forward(img)
                    _, _, _, predict_prob_t = classifier_t.forward(feature)

                    counter.addOneBatch(variable_to_numpy(predict_prob_t), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()
            
            print('>>>>>>>>>>>accuracy>>>>>>>>>>>>>>>>.')
            print(acc_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
            if best_acc < acc_test:
                best_acc = acc_test
                model_dict = {
                        'global_step': global_step + 1,
                        'state_dict': trainable_tragetNet.state_dict(),
                        'accuracy': acc_test}
                # torch.save(model_dict, join('/home1/wangfan/Projectt/SFDAIB/pretrained_weights/'+str(args.data.dataset.source) + str(args.data.dataset.target) +'/' + 'domain'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'accBEST_model_checkpoint2.pth.tar'))


exit()
