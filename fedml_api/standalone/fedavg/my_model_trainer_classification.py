import logging
import copy
import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, test_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        client_result = {'train_losses':[], 'test_losses':[], 'train_acc':[], 'test_acc':[], 'best_acc':-1}
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
            train_local_metrics = self.test(train_data, device, args)
            train_num_samples = copy.deepcopy(train_local_metrics['test_total'])
            train_num_correct = copy.deepcopy(train_local_metrics['test_correct'])
            train_losses = copy.deepcopy(train_local_metrics['test_loss'])

            test_local_metrics = self.test(test_data, device, args)
            test_num_samples = copy.deepcopy(test_local_metrics['test_total'])
            test_num_correct = copy.deepcopy(test_local_metrics['test_correct'])
            test_losses = copy.deepcopy(test_local_metrics['test_loss'])

            # test on training dataset
            train_acc = train_num_correct / train_num_samples
            train_loss = train_losses / train_num_samples

            # test on test dataset
            test_acc = test_num_correct / test_num_samples
            test_loss = test_losses / test_num_samples

            if client_result['best_acc'] < test_acc:
                client_result['best_acc'] = test_acc

            client_result['train_losses'].append(train_loss)
            client_result['train_acc'].append(train_acc)

            client_result['test_losses'].append(test_loss)
            client_result['test_acc'].append(test_acc)
        return client_result

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
