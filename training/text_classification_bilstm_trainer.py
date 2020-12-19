import logging

import torch

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class TextClassificationBiLSTMTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.epochs):
            batch_loss = []
            batch_acc = []
            for batch_idx, batch_data in enumerate(train_data):
                x = torch.tensor(batch_data["X"])
                y = torch.tensor(batch_data["Y"])
                seq_lens = torch.tensor(batch_data["seq_lens"])
                if device is not None:
                    x = x.to(device=device)
                    y = y.to(device=device)
                    seq_lens = seq_lens.to(device=device)
                optimizer.zero_grad()
                prediction = model(x, x.size()[0], seq_lens, device)
                loss = criterion(prediction, y)
                num_corrects = torch.sum(torch.argmax(prediction, 1).eq(y))
                acc = 100.0 * num_corrects / x.size()[0]
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('(Trainer_ID {}. Local Training Epoch: {} '
                            '\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss),
                                                                    sum(epoch_acc) / len(epoch_acc)))

    def test(self, test_data, device, args):
        model = self.model

        model.eval()
        model.to(device)

        test_loss = test_acc = test_total = 0.
        criterion = torch.nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_data):
                x = torch.tensor(batch_data["X"])
                y = torch.tensor(batch_data["Y"])
                seq_lens = torch.tensor(batch_data["seq_lens"])
                if device is not None:
                    x = x.to(device=device)
                    y = y.to(device=device)
                    seq_lens = seq_lens.to(device=device)

                prediction = model(x, x.size()[0], seq_lens, device)
                loss = criterion(prediction, y)
                num_corrects = torch.sum(torch.argmax(prediction, 1).eq(y))

                test_acc += num_corrects.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_total, test_loss
