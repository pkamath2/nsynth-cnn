import pandas as pd
import torch
import sklearn

import util

class Train_And_Test():
    def __init__(self, total_epoch, input_shape, model, optimizer, loss, device, train_loader, test_loader, validate_loader):
        self.total_epoch = total_epoch
        self.input_shape = input_shape
        self.current_epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validate_loader = validate_loader
        self.device = device
        
        
    def train(self, epoch):
        self.current_epoch = epoch
        training_loss = 0.0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.float()
            data = data.to(self.device)
            target = target.to(self.device)

            data = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            self.optimizer.zero_grad()
            output = self.model(data)

            running_loss = self.loss(output, target)
            training_loss += running_loss.item()
            running_loss.backward()
            self.optimizer.step()
        
            if batch_idx % 2 == 0:
                    print('Train Epoch: {}, Batch id: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), running_loss.item()))
        print(training_loss)
        # To improve legibility only one loss norm is plotted for an epoch (Goodfellow - pg 276 diag 8.1)
        training_loss /= len(self.train_loader)
        print(training_loss)
        return training_loss

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data = data.float()
            data, target = data.to(self.device), target.to(self.device)

            data = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            self.optimizer.zero_grad()
            output = self.model(data)
            test_loss += self.loss(output, target).item() # sum up batch loss  
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability     
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            accuracy))
        return test_loss, accuracy

    def validate(self):
        self.model.eval()
        validate_loss = 0
        correct = 0
        for data, target in self.validate_loader:
            data = data.float()
            data, target = data.to(self.device), target.to(self.device)

            data = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            output = self.model(data)
            validate_loss += self.loss(output, target).item() # sum up batch loss  
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability     
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        validate_loss /= len(self.validate_loader)
        accuracy = 100. * correct / len(self.validate_loader.dataset)
        print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validate_loss, correct, len(self.validate_loader.dataset),
            accuracy))
        return validate_loss, accuracy

    def save_model(self, model_location):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, model_location) #'models/cnn-nsynth-3class-classification-keyboard-guitar-bass')
        print('Saved Model')

    def build_cm(self):
        self.model.eval()
        test_loss = 0
        running_correct = []
        target_correct = []
        correct = 0
        for data, target in self.test_loader:
            data = data.float()
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            output = self.model(data)
            test_loss += self.loss(output, target).item() # sum up batch loss  
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability  
            c = pred.eq(target.data.view_as(pred)).cpu().sum().item()
            running_correct.append(pred.cpu().flatten().tolist())
            target_correct.append(target.cpu().flatten().tolist())
            correct += c

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print('\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            accuracy))
        running_correct = [item for itemlist in running_correct for item in itemlist]
        target_correct = [item for itemlist in target_correct for item in itemlist]
        
        cm = sklearn.metrics.confusion_matrix(target_correct, running_correct)
        return cm

    def load_model(self, model_location, device):
        checkpoint = torch.load(model_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        return self.model
