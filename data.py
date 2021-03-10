from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import csv
import random
import pandas as pd



def loaddata():
    source_classes = [i for i in range(args.data.dataset.n_total)]
    # print(source_classes)
    target_classes = [i for i in range(args.data.dataset.n_share)]
    # print(target_classes)
    domains = ['amazon', 'dslr', 'webcam']
    domains1 = ['Art','Clipart','Product','Real_world']
    domains2 = ['simulation','real']

    #先利用随机数划分成不同的txt文件
    train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        
    ])
    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),

    ])

    target_train_transform = Compose([
        Resize(256),
        RandomCrop(224),
        ToTensor(),
    ])
    target_test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
    ])
    source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=train_transform, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                                transform=test_transform, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=target_train_transform, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                transform=target_test_transform , filter=(lambda x: x in target_classes))

    classes = source_train_ds.labels
    freq = Counter(classes)
    class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

    source_weights = [class_weight[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))
    #source 和 test samplers 是随机进行替换的抽取的，source 不知道 label,但是test 是知道label的
    source_train_dl =DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                                sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)
    target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                                num_workers=args.data.dataloader.data_workers, drop_last=True)
    target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                                num_workers=1, drop_last=False)

    csv_path =''
    csv_path1 = ''
    csv_path2 = ''
    txtPath = ''
    txtPath1 = ''
    len_file = 0
    #表示数据集是office
    if(args.data.dataset.name == 'office'): 
        csv_path = args.data.dataset.root_path+'/'+domains[args.data.dataset.source]+'_list.csv'
        csv_path1 = args.data.dataset.root_path+'/'+domains[args.data.dataset.source]+'_list_s.csv'
        csv_path2 = args.data.dataset.root_path+'/'+domains[args.data.dataset.source]+'_list_s_t.csv'
        txtPath = args.data.dataset.root_path+'/'+domains[args.data.dataset.source]+'_list_s.txt' 
        txtPath1 = args.data.dataset.root_path+'/'+domains[args.data.dataset.source]+'_list_s_t.txt'
    #表示数据集是officehome
    elif(args.data.dataset.name == 'officehome'):
        csv_path = args.data.dataset.root_path+'/'+domains1[args.data.dataset.source]+'_list.csv'
        csv_path1 = args.data.dataset.root_path+'/'+domains1[args.data.dataset.source]+'_list_s.csv'
        csv_path2 = args.data.dataset.root_path+'/'+domains1[args.data.dataset.source]+'_list_s_t.csv'
        txtPath = args.data.dataset.root_path+'/'+domains1[args.data.dataset.source]+'_list_s.txt' 
        txtPath1 = args.data.dataset.root_path+'/'+domains1[args.data.dataset.source]+'_list_s_t.txt' 
    elif(args.data.dataset.name == 'visda'):
        csv_path = args.data.dataset.root_path+'/'+domains2[args.data.dataset.source]+'_list.csv'
        csv_path1 = args.data.dataset.root_path+'/'+domains2[args.data.dataset.source]+'_list_s.csv'
        csv_path2 = args.data.dataset.root_path+'/'+domains2[args.data.dataset.source]+'_list_s_t.csv'
        txtPath = args.data.dataset.root_path+'/'+domains2[args.data.dataset.source]+'_list_s.txt' 
        txtPath1 = args.data.dataset.root_path+'/'+domains2[args.data.dataset.source]+'_list_s_t.txt'  
    with open(csv_path) as f:
        len_file = len(f.readlines())
        #取随机数
    num = int(len_file*0.1)
    #非重复的随机数
    random_index = random.sample(range(0,len_file-1),num)
    #csv文件复制
    csvFile = open(csv_path1,'w',newline='',encoding='utf-8')
    writer = csv.writer(csvFile)
    csvFile1 = open(csv_path2,'w',newline='',encoding='utf-8')
    writer1 = csv.writer(csvFile1)

    reader = pd.read_csv(csv_path, encoding='utf-8',header=None)
    index = 0
    for row in reader.values:
        if(index not in random_index):
            writer.writerow(row)
        else:
            writer1.writerow(row) 
        index = index+1 
    csvFile.close()
    csvFile1.close()

    data_s = pd.read_csv(csv_path1, encoding='utf-8',header=None)
            
    with open(txtPath,'w', encoding='utf-8') as f:
        for line in data_s.values:
            f.write((str(line[0])+'\t'+str(line[1])+'\n'))
    #注意读文件时候去掉表头
    data_s1 = pd.read_csv(csv_path2, encoding='utf-8',header=None)
            
    with open(txtPath1,'w', encoding='utf-8') as f:
        for line in data_s1.values:
            f.write((str(line[0])+'\t'+str(line[1])+'\n'))

    os.remove(csv_path1)
    os.remove(csv_path2)
    source_train_dds = FileListDataset(list_path=txtPath, path_prefix=dataset.prefixes[args.data.dataset.source],
                        transform=train_transform, filter=(lambda x: x in source_classes))
    source_train_ddl =DataLoader(dataset=source_train_dds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                        num_workers=args.data.dataloader.data_workers, drop_last=True)
                                    
    source_test_dds = FileListDataset(list_path=txtPath1,path_prefix=dataset.prefixes[args.data.dataset.source],
                        transform=test_transform, filter=(lambda x: x in source_classes))
    source_test_ddl = DataLoader(dataset=source_test_dds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
    return source_train_ddl,source_test_ddl,target_train_dl,target_test_dl,source_train_dl,source_test_dl