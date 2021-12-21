import copy
import logging
import random
import pickle

import numpy as np
import torch

from fedml_api.standalone.fedavg.client import Client
from utils.results import *
import time


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, results_class, save_dir):
        self.device = device
        self.args = args
        self.results_class = results_class
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.class_num = class_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.save_dir = save_dir
        self.start = time.time()

        self.model_trainer = model_trainer

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    # 참여클라이언트 더미 생성, 매라운드마다 client list에 있는 client의 값을 바꿈
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        best_acc = 0
        best_round = 0
        best_w = 0
        self.results_class.init_server_results(self.args.client_num_in_total)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))
            self.start = time.time()

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """

            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)

            logging.info("client_indexes = " + str(client_indexes))

            each_client_result = {}
            participants_idx = []
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w, client_result = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                each_client_result[client_idx] = client_result
                participants_idx.append(client_idx)

            self.results_class.client_participants_results[round_idx] = each_client_result

            # update global weights
            w_global = self._aggregate(w_locals)
            # update global weight to all client
            self.model_trainer.set_model_params(w_global)

            # test results
            self._local_test_on_all_clients(round_idx, participants_idx, self.args.comm_round - 1)
            test_acc = self.results_class.server_results['test_acc'][-1]

            if best_acc < test_acc:
                best_acc = test_acc
                best_w = w_global
                best_round = round_idx

            logging.info("################running time: {}".format(time.time()-self.start))

        torch.save(best_w, self.save_dir + '/best_modelweight_round{}_acc{}.pt'.format(round_idx,round(test_acc,3)))
        ## Save pickle
        results_dict = {'server_results' : self.results_class.server_results, 'client_idx_results' : self.results_class.client_idx_results,
                        'client_participants_results' : self.results_class.client_participants_results, 'data_local_cls_dict' : self.results_class.data_local_cls_dict  }

        with open(self.save_dir + "/results.pickle","wb") as fw:
            pickle.dump(results_dict, fw)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx, participants_idx, last_round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
                    'num_samples': [],
                    'num_correct': [],
                    'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]


        for client_idx in range(self.args.client_num_in_total):

            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            # test on training dataset
            train_acc = train_local_metrics['test_correct'] / train_local_metrics['test_total']
            train_loss = train_local_metrics['test_loss'] / train_local_metrics['test_total']

            # test on test dataset
            test_acc = test_local_metrics['test_correct'] / test_local_metrics['test_total']
            test_loss = test_local_metrics['test_loss'] / test_local_metrics['test_total']

            self.results_class.client_idx_results[client_idx]['train_losses'].append(train_loss)
            self.results_class.client_idx_results[client_idx]['test_losses'].append(test_loss)
            self.results_class.client_idx_results[client_idx]['train_acc'].append(train_acc)
            self.results_class.client_idx_results[client_idx]['test_acc'].append(test_acc)


        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        self.results_class.server_results['train_losses'].append(train_loss)
        self.results_class.server_results['train_acc'].append(train_acc)
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        self.results_class.server_results['test_losses'].append(test_loss)
        self.results_class.server_results['test_acc'].append(test_acc)
        logging.info(stats)
