import json
import logging
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import MNIST_truncated


def _data_transforms_mnist():
    #CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    #CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    #train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def get_dataloader(X_train, y_train, X_test, y_test, train_bs, test_bs, train_dataidxs=None, test_dataidxs=None):
    dl_obj = MNIST_truncated
    transform_train, transform_test = _data_transforms_mnist()

    train_ds = dl_obj(X_train, y_train, dataidxs=train_dataidxs, transform=transform_train)
    test_ds = dl_obj(X_test, y_test, dataidxs=test_dataidxs, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl

def load_mnist_data(train_path, test_path):
    train_files = os.listdir(train_path)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_path, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        X_train = []
        y_train = []
        for id in cdata['user_data'].keys():
            for x in cdata['user_data'][id]['x']:
                X_train.append(x)
            for y in cdata['user_data'][id]['y']:
                y_train.append(y)
        X_train = np.array(X_train)
        X_train = X_train.reshape(-1, 28, 28)
        y_train = np.array(y_train)
        #train_data = train_data.update(cdata['user_data'])

    test_files = os.listdir(test_path)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_path, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        X_test = []
        y_test = []
        for id in cdata['user_data'].keys():
            for x in cdata['user_data'][id]['x']:
                X_test.append(x)
            for y in cdata['user_data'][id]['y']:
                y_test.append(y)
        X_test = np.array(X_test)
        X_test = X_test.reshape(-1, 28, 28)
        y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

def record_net_data_stats(y_data, net_dataidx_map):
    data_cls_dict = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_data[dataidx], return_counts=True)
        tmp = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}
        data_cls_dict[net_i] = tmp
    #logging.debug('Data statistics: %s' % str(net_cls_counts))
    return data_cls_dict

def partition_data(train_path, test_path, partition, n_nets, alpha, batch_size):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(train_path, test_path)
    n_train = X_train.shape[0]
    class_num = len(np.unique(y_train))
    local_data_cls_dict = {}
    # n_test = X_test.shape[0]

    if partition == "homo":
        for idx, y_phase in enumerate([y_train, y_test]):
            K = class_num
            N = y_phase.shape[0]
            logging.info("N = " + str(N))
            net_dataidx_map = {}
            idx_batchs = [[] for _ in range(n_nets)]

            data_num_list = []

            for k in range(K):
                idx_k = np.where(y_phase == k)[0]
                data_num_list.append(idx_k.shape[0])
            min_datanum = min(data_num_list)

            for k in range(K):
                idx_k = np.where(y_phase == k)[0]
                idx_k = np.random.choice(idx_k, min_datanum, replace=False)
                np.random.shuffle(idx_k)

                class_data_num = idx_k.shape[0] // n_nets
                for i in range(n_nets):
                    idx_batchs[i] = idx_batchs[i] + list(idx_k[i*class_data_num:(i+1)*class_data_num])
                    if (i+2)*class_data_num > idx_k.shape[0]:
                        break

            for j in range(n_nets):
                if len(idx_batchs[j]) == 0:
                    logging.info("*********too many client // there is client to get 0 of datas ***************")
                    exit()
                np.random.shuffle(idx_batchs[j])
                net_dataidx_map[j] = idx_batchs[j]
            data_cls_dict = record_net_data_stats(y_phase, net_dataidx_map)

            if idx == 0:
                train_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys] = {'train': value}
            else:
                test_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys]['test'] = value

    elif partition == "hetero":
        for idx, y_phase in enumerate([y_train, y_test]):
            min_size = 0
            K = class_num
            N = y_phase.shape[0]
            logging.info("N = " + str(N))
            net_dataidx_map = {}

            while min_size < 10:
                idx_batchs = [[] for _ in range(n_nets)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_phase == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batchs)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batchs = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batchs, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batchs])

            for j in range(n_nets):
                if len(idx_batchs[j]) == 0:
                    logging.info("*********too many client // there is client to get 0 of datas ***************")
                    exit()
                np.random.shuffle(idx_batchs[j])
                net_dataidx_map[j] = idx_batchs[j]
            data_cls_dict = record_net_data_stats(y_phase, net_dataidx_map)

            if idx == 0:
                train_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys] = {'train': value}
            else:
                test_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys]['test'] = value

    elif partition == "class":
        if n_nets == 100:
            piece = 20
        elif n_nets == 1000:
            piece = 200
        else:
            print("Error : only num of Total_client is 100 or 1000!! ")
            exit()

        # 각 클래스별로 조각이 들어갈 클라이언트 인덱스 생성 (client_idxs)
        choice_classnum = 2
        total_client_idxs = [i for i in range(n_nets)] * choice_classnum
        np.random.shuffle(total_client_idxs)

        client_idxs_list = []
        while True:
            dummy_idx = []
            break_num = 0 # 클라이언트 선택시 마지막 더미에서 같은 클라이언트가 생기는 경우가 발생하여서, 이럴 경우 무한루프가 돈다. 따라서 무한루프 확인용으로 해당 변수를 사용한다.
            while True:
                break_num+=1

                idx = total_client_idxs.pop()
                if idx not in dummy_idx:
                    dummy_idx.append(idx)
                    break_num = 0
                else:
                    total_client_idxs.insert(0,idx)

                if break_num > piece+1:
                    # 현재 더미리스트에선 중복방지를 할 수 없기 때문에 다른 리스트의 값과 교환을 해야한다.
                    # 1. 다른 리스트에서 현재 중복된 값이 있는지 확인한다.
                    # 2. 없다면 해당 리스트에서 현재 더미리스트에 없는 값을 추출한다.
                    # 3. 서로 교환한다.
                    idx = total_client_idxs.pop()
                    change_check = False
                    for i, client_idxs in enumerate(client_idxs_list):
                        if idx not in client_idxs:
                            for client_idx in client_idxs:
                                if client_idx not in dummy_idx:
                                    change_check = True
                                    client_idxs_list[i].append(idx)
                                    client_idxs_list[i].remove(client_idx)
                                    dummy_idx.append(client_idx)
                                    break
                        if change_check == True:
                            break
                    break_num = 0
                if len(dummy_idx) == piece:
                    break

            client_idxs_list.append(dummy_idx)

            if len(client_idxs_list) == class_num:
                break

        proportions_list = []
        for _ in range(class_num):
            proportions_list.append(np.random.dirichlet(np.repeat(alpha, piece)))
        for idx, y_phase in enumerate([y_train, y_test]):
            K = class_num
            N = y_phase.shape[0]
            logging.info("N = " + str(N))
            net_dataidx_map = {}
            idx_batchs = [[] for _ in range(n_nets)]


            for k in range(K):
                idx_k = np.where(y_phase == k)[0]
                np.random.shuffle(idx_k)

                class_data_num = idx_k.shape[0]
                class_data_num = class_data_num - piece
                proportions = list(map(int,((np.floor(proportions_list[k] * class_data_num)) + 1))) # 계산의 편의를 위해 이전 라인에서 클래스 개수*2만큼 뺀다음, 계산된 값을 내림 후 1을 더 하는 방식을 취한다.
                print(sum(proportions))
                print(idx_k.shape[0])

                idx_k = list(idx_k)
                for i, client_idx in enumerate(client_idxs_list[k]):
                    dummy_idx = []
                    for _ in range(proportions[i]):
                        dummy_idx.append(idx_k.pop())
                    idx_batchs[client_idx] = idx_batchs[client_idx] + dummy_idx

            for j in range(n_nets):
                if len(idx_batchs[j]) == 0:
                    logging.info("*********too many client // there is client to get 0 of datas ***************")
                    exit()
                np.random.shuffle(idx_batchs[j])
                net_dataidx_map[j] = idx_batchs[j]
            data_cls_dict = record_net_data_stats(y_phase, net_dataidx_map)

            if idx == 0:
                train_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys] = {'train': value}
            else:
                test_net_dataidx_map = net_dataidx_map
                for keys, value in data_cls_dict.items():
                    local_data_cls_dict[keys]['test'] = value

    return X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map, local_data_cls_dict, class_num

def load_partition_data_mnist_custom(batch_size,
                              train_path="./../../../data/MNIST/train",
                              test_path="./../../../data/MNIST/test",
                              partition_method='hetero',
                              client_num=10,
                              partition_alpha=0.5
                              ):
        X_train, y_train, X_test, y_test, train_net_dataidx_map, test_net_dataidx_map, local_data_cls_dict, class_num = partition_data(train_path,
                                                                                                                                                     test_path,
                                                                                                                                                     partition_method,
                                                                                                                                                     client_num,
                                                                                                                                                     partition_alpha,
                                                                                                                                                     batch_size)
        #logging.info("traindata_cls_counts = " + str(train_data_local_cls_dict))
        train_data_num = sum([len(train_net_dataidx_map[r]) for r in range(client_num)])

        train_data_global, test_data_global = get_dataloader(X_train, y_train, X_test, y_test, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        test_data_num = len(test_data_global)

        # get local dataset
        train_data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(client_num):
            train_dataidxs = train_net_dataidx_map[client_idx]
            test_dataidxs = test_net_dataidx_map[client_idx]
            train_local_data_num = len(train_dataidxs)
            test_local_data_num = len(test_dataidxs)
            logging.info("client_idx = %d, train_local_sample_number = %d, test_local_sample_number = %d  " % (client_idx, train_local_data_num, test_local_data_num))

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = get_dataloader(X_train, y_train, X_test, y_test, batch_size, batch_size,
                                                     train_dataidxs, test_dataidxs)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local)))
            train_data_local_num_dict[client_idx] = len(train_data_local)
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local

        return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, local_data_cls_dict, class_num
