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
    def __init__(self, dataset, device, args, model_trainer,save_dir):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.save_dir = save_dir
        self.start = time.time()

        self.model_trainer = model_trainer
        self.server_results = {'train_losses':[], 'test_losses':[], 'train_acc':[], 'test_acc':[] }
        self.client_results = {}

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

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

            each_client_result = []
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
                each_client_result.append(client_result)

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            self.client_results[round_idx] = each_client_result
            test_acc = self._local_test_on_all_clients(round_idx)
            if best_acc < test_acc:
                best_acc = test_acc
                best_w = w_global
                best_round = round_idx

            logging.info("################running time: {}".format(time.time()-self.start))
        draw_acc(self.server_results['train_acc'], self.server_results['test_acc'], self.save_dir, 'server')
        draw_losses(self.server_results['train_losses'], self.server_results['test_losses'], self.save_dir, 'server')
        draw_clientresult(self.client_results, best_round, self.save_dir, 3)
        torch.save(best_w, self.save_dir + '/bestmodel_state_dict_round{}_acc{}.pt'.format(round_idx,round(test_acc,3)))

        ## Save pickle
        with open(self.save_dir + "/client_server_results.pickle","wb") as fw:
            pickle.dump(self.client_results, fw)
            pickle.dump(self.server_results, fw)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

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

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))


        client = self.client_list[0]
        client.update_local_dataset(0, self.train_global,
                                    self.test_global,
                                    self.train_data_num_in_total + self.test_data_num_in_total)

        # train data
        train_server_metrics = client.local_test(False)
        train_num_samples = copy.deepcopy(train_server_metrics['test_total'])
        train_num_correct = copy.deepcopy(train_server_metrics['test_correct'])
        train_losses = copy.deepcopy(train_server_metrics['test_loss'])

        # test data
        test_server_metrics = client.local_test(True)
        test_num_samples = copy.deepcopy(test_server_metrics['test_total'])
        test_num_correct = copy.deepcopy(test_server_metrics['test_correct'])
        test_losses = copy.deepcopy(test_server_metrics['test_loss'])

        # test on training dataset
        train_acc = train_num_correct / train_num_samples
        train_loss = train_losses / train_num_samples

        # test on test dataset
        test_acc = test_num_correct / test_num_samples
        test_loss = test_losses / test_num_samples

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        self.server_results['train_losses'].append(train_loss)
        self.server_results['train_acc'].append(train_acc)
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        self.server_results['test_losses'].append(test_loss)
        self.server_results['test_acc'].append(test_acc)
        logging.info(stats)

        return test_acc


    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
