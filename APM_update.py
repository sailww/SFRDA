from data import *
from lib import  *
import time
import numpy as np
import torch
from scipy.spatial.distance import cosine

#H2
T = 1
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

def calculate_entropy(max_entropy):
    entropy_re = - max_entropy * np.log(max_entropy + 1e-10)
    return entropy_re
def getmax_secondmax(x):
    top_k_value = np.sort(x)
    return top_k_value[-1] , top_k_value[-2]

def calculate_entropysecond(second_max_entropy):
    return  - second_max_entropy*np.log(second_max_entropy)

def calculate_natigative_logit(list_val):
    total = 0
    for ele in range(0, len(list_val)):
        total = total + list_val[ele]
    return  T *10* np.log(total)
def calculate_avg_entropy(h_dict,available_cls):
    sub_total_size = 0 
    sub_total_entropy = 0.0
    avg_entropy = 0.0
    h_dict_entropy = {}
    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        h_ava_size = len(ents_np)
        ent_idxs = np.argsort(ents_np)
        sub_total_size = sub_total_size + h_ava_size
        sub_total_cls_entropy =0.0
        for cls1 in range(h_ava_size):
            sub_total_entropy = sub_total_entropy + ents_np[ent_idxs[cls1]]
            sub_total_cls_entropy = sub_total_cls_entropy + ents_np[ent_idxs[cls1]]
        h_dict_entropy[cls] = sub_total_cls_entropy /h_ava_size      
    avg_entropy = sub_total_entropy / sub_total_size
    return h_dict_entropy,avg_entropy




def APM_init_update(feature_extractor, classifier_t,p,r,target_train_dl,epoch_id):
    start_time = time.time()
    available_cls = []
    h_dict = {}
    h_dict_energy = {}
    h_dict_flag = {}
    h_dict_avg = {}
    feat_dict = {}
    h_dict_true_label = {}
    missing_cls = []
    select_class_entropy = {}
    select_class_energy = {}
    after_softmax_numpy_for_emergency = []
    feature_numpy_for_emergency = []
    max_prototype_bound = 100
    


    for cls in range(args.data.dataset.n_share):
        h_dict[cls] = []
        feat_dict[cls] = []
        h_dict_flag[cls] = 0
        h_dict_energy[cls] = []
        h_dict_avg[cls] = []
        select_class_entropy[cls] = []
        select_class_energy[cls] = []
        h_dict_true_label[cls] = []

    for (im_target_lbcorr, label_target_lbcorr) in target_train_dl:
        im_target_lbcorr = im_target_lbcorr.cuda()
        fc1_lbcorr = feature_extractor.forward(im_target_lbcorr)
        _, _, logit_t, after_softmax = classifier_t.forward(fc1_lbcorr)
        after_softmax_numpy_for_emergency.append(after_softmax.data.cpu().numpy())
        feature_numpy_for_emergency.append(fc1_lbcorr.data.cpu().numpy())

        pseudo_label = torch.argmax(after_softmax, dim=1)
        pseudo_label = pseudo_label.cpu()
        
       
        
        #测试H1(p^2)或测试 H2
        # 转成array
        after_softmax_array = after_softmax.detach().cpu().numpy()
        #每行的最大值，最值得相信的那个值
        after_softmax_array = after_softmax_array.max(axis = 1 )
        list_max = [calculate_entropyH2(x) for _,x in enumerate(after_softmax_array)]
        list_max = list_max / np.log(after_softmax.size(1))
        list_max = torch.Tensor(list_max)
        list_max = list_max.cpu()

         ##logit_t用于计算一批图片的能量
         ##利用能量将分布内和分布外的样本进行筛选，将0.8*能量之外的能量作为ood样本看待，然后只取一个
         # try一下，不可以站在上帝视角来看待问题
        
        logit_t_energy = logit_t.detach().cpu().numpy()
        logit_t_energy = logit_t_energy/T
        list_logit = [np.exp(x)/10  for _,x in enumerate(logit_t_energy)]
        # -E(X)  值越大，表示其越是分布内的样本，否则表示其越是分布外的样本
        energy = [calculate_natigative_logit(x) for _,x in enumerate(list_logit)]
        energy = energy / np.log(after_softmax.size(1)) 
        energy = torch.Tensor(energy)
        energy = energy.cpu()
 

        for cls in range(args.data.dataset.n_share):
            # stack H for each class
            cls_filter = (pseudo_label == cls)
            list_loc = cls_filter.numpy().tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            list_loc = list_loc.long()
            # num_element = list(list_loc.data.numpy())
            if len(list_loc) == 0:
                missing_cls.append(cls)
                continue
            available_cls.append(cls)
            filtered_ent = torch.gather(list_max, dim=0, index=list_loc)
            filtered_energy = torch.gather(energy, dim=0, index=list_loc)
            filtered_feat = torch.gather(fc1_lbcorr.cpu(), dim=0, index=list_loc.unsqueeze(1).repeat(1, 2048))
            filtered_true = torch.gather(label_target_lbcorr, dim=0, index=list_loc)

            h_dict[cls].append(filtered_ent.cpu().data.numpy())
            feat_dict[cls].append(filtered_feat.cpu().data.numpy())
            h_dict_energy[cls].append(filtered_energy.cpu().data.numpy())
            h_dict_true_label[cls].append(filtered_true)

    available_cls = np.unique(available_cls)

    prototype_memory = []
    prototype_memory_dict = {}
    after_softmax_numpy_for_emergency = np.concatenate(after_softmax_numpy_for_emergency, axis=0)
    feature_numpy_for_emergency = np.concatenate(feature_numpy_for_emergency, axis=0)

    class_protypeNum_dict = {}
    max_prototype = 0
    
    ss1 , avg_filter_entropy = calculate_avg_entropy(h_dict,available_cls) 
    ss2 , avg_filter_energy = calculate_avg_entropy(h_dict_energy,available_cls)




    select_item = 0
    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ents_np1 = np.concatenate(h_dict_energy[cls], axis=0)
        ents_np2 = np.concatenate(h_dict_true_label[cls], axis=0)
        total = 0
        for index in range(len(ents_np)):
            if(ents_np[index] <= ss1[cls] * p and ents_np1[index] >= ss2[cls] * r):
                total = total +1
                # if(ents_np2[index] == cls):
                #     select_item +=1
                #确定当前这个是正确的如果是就++
        class_protypeNum_dict[cls] = total
    if max_prototype < class_protypeNum_dict[cls]:
        max_prototype = class_protypeNum_dict[cls]

    for cls in available_cls:
        if(class_protypeNum_dict[cls] == 0):
            class_protypeNum_dict[cls] = 1 
  
    if max_prototype > 100:
        max_prototype = max_prototype_bound


    #当前样本的平均熵
    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ent_idxs = np.argsort(ents_np)
        len_e = class_protypeNum_dict[cls]
        sub_entropy = 0.0
        for cls1 in range(len_e):
            sub_entropy += ents_np[ent_idxs[cls1]]
        select_class_entropy[cls] = sub_entropy/len_e 
    #当前样本的平均能量
    for cls in available_cls:
        ents_np = np.concatenate(h_dict_energy[cls], axis=0)
        ent_idxs = np.argsort(ents_np)
        len_e = class_protypeNum_dict[cls]
        sub_energy = 0.0
        for cls1 in range(len_e):
            sub_energy += ents_np[ent_idxs[cls1]]
        select_class_energy[cls] = sub_energy/len_e 
    # if(epoch_id %2 == 0):
    #     #计算挑选的准确率
    #     total_num = 0.0
    #     total_accuracy = 0.0
    #     for cls in available_cls:
    #         len_e = class_protypeNum_dict[cls]
    #         # ents_np = np.concatenate(h_dict_true_label[cls], axis=0)
    #         total_num += len_e
    #     acc_test = select_item / total_num
    #     print("<<<<<<<<<<<<<select accuracy and detail information<<<<<<<<<<<<<")
    #     print(epoch_id)
    #     print(select_item)
    #     print(total_num)
    #     print(acc_test)
    #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        
    for cls in range(args.data.dataset.n_share):

        if cls in available_cls:
            ents_np = np.concatenate(h_dict[cls], axis=0)
            feats_np = np.concatenate(feat_dict[cls], axis=0)
            ent_idxs = np.argsort(ents_np)

            truncated_feat = feats_np[ent_idxs[:class_protypeNum_dict[cls]]]
            fit_to_max_prototype = np.concatenate([truncated_feat] * (int(max_prototype / truncated_feat.shape[0]) + 1),
                                                  axis=0)
            fit_to_max_prototype = fit_to_max_prototype[:max_prototype, :]

            prototype_memory.append(fit_to_max_prototype)
            prototype_memory_dict[cls] = fit_to_max_prototype
        else:
            after_softmax_torch_for_emergency = torch.Tensor(after_softmax_numpy_for_emergency)
            emergency_idx = torch.argsort(after_softmax_torch_for_emergency, descending=True, dim=1)
            cls_emergency_idx = emergency_idx[:, cls]
            cls_emergency_idx = cls_emergency_idx[0]
            cls_emergency_idx_numpy = cls_emergency_idx.data.numpy()

            copied_features_emergency = np.concatenate(
                [np.expand_dims(feature_numpy_for_emergency[cls_emergency_idx_numpy], axis=0)] * max_prototype, axis=0)

            prototype_memory.append(copied_features_emergency)
            prototype_memory_dict[cls] = copied_features_emergency

    print("** APM update... time:", time.time() - start_time)
    prototype_memory = np.concatenate(prototype_memory, axis=0)
    num_prototype_ = int(max_prototype)

    return prototype_memory, num_prototype_, prototype_memory_dict,select_class_entropy,select_class_energy
