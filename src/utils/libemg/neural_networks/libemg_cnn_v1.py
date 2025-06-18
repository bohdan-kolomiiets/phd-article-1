import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from libemg.datasets import *

from utils.early_stopping import EarlyStopping
from utils.model_checkpoint import ModelCheckpoint

class CNN_V1(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256, generator=None):
        super().__init__()

        self.generator = generator

        self.n_output = n_output

        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels # 10
        l1_filters = n_filters # 64
        l2_filters = n_filters // 2 # 32
        l3_filters = n_filters // 4 # 16
        
        # setup layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(l0_filters, l1_filters, kernel_size=5), # Size([batches, 10, 200]) -> Size([batches, 64, 196])
            nn.BatchNorm1d(l1_filters), # compute mean and variance over all elements and apply normalization, for each of the 64 output channels 
            nn.ReLU(),
            nn.Conv1d(l1_filters, l2_filters, kernel_size=5), # Size([batches, 64, 196]) -> Size([batches, 32, 192])
            nn.BatchNorm1d(l2_filters),
            nn.ReLU(),
            nn.Conv1d(l2_filters, l3_filters, kernel_size=5),  # Size([batches, 32, 192]) -> Size([batches, 16, 188])
            nn.BatchNorm1d(l3_filters),
            nn.ReLU()
        )

        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.convolutional_layers(example_input) # Size([1, 16, 188])
        size_after_conv = conv_output.view(-1).shape[0] # 16 * 188 = 3008
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, self.n_output) # fully connected layer that transforms 3008 inputs to 11 gesture classes
        
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)

        CNN_V1.initialize_with_glorot_weight_zero_bias(self, generator)

        self = CNN_V1._try_move_to_accelerator(self)
        

    def forward(self, x):
        x = self.convolutional_layers(x) # Size([64, 10, 200]) -> Size([64, 16, 188])
        x = x.view(x.shape[0],-1) # Size([64, 16, 188]) -> Size([64, 3008])
        # x = self.act(x) # fix: redundant
        x = self.output_layer(x) # Size([64, 3008]) -> Size([64, 11])
        # x = self.softmax(x) # fix: incorrect, CrossEntropyLoss expects raw logits from the linear layer as internally it does softmax calculations by itself, otherwise leads to tiny gradients and slow training
        return x

    @staticmethod
    def _try_move_to_accelerator(obj):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.accelerator.is_available():
            obj = obj.to(torch.accelerator.current_accelerator())
        return obj
    
    # TODO: each module has modules() inside, you can make it static and pass any layer to initialize it this way
    # TODO: during training try to call this method with layer passed and check if it will reset it 
    # TODO: compare with calling module.reset_parameters() before calling glorot_weight_zero_bias
    # TODO: rename this emthod?
    @staticmethod
    def initialize_with_glorot_weight_zero_bias(module, generator=None):
        """
        Based on
        https://robintibor.github.io/braindecode/source/braindecode.torch_ext.html#module-braindecode.torch_ext.init
        Initalize parameters of all modules by initializing weights with glorot uniform/xavier initialization, and setting biases to zero. Weights from batch norm layers are set to 1.
        """
        for sum_module in module.modules():
            if hasattr(sum_module, "weight"):
                if not ("BatchNorm" in sum_module.__class__.__name__):
                    nn.init.xavier_uniform_(sum_module.weight, gain=1, generator=generator)
                else:
                    nn.init.constant_(sum_module.weight, 1)
            if hasattr(sum_module, "bias"):
                if sum_module.bias is not None:
                    nn.init.constant_(sum_module.bias, 0)


    def fit(self, dataloader_dictionary, num_epochs, adam_learning_rate, adam_weight_decay, verbose, model_checkpoint: ModelCheckpoint, training_log_callback=None):

        early_stopping = EarlyStopping(patience=4, acceptable_delta=0.03, verbose=True)

        optimizer = optim.Adam(self.parameters(), lr=adam_learning_rate, weight_decay=adam_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=adam_learning_rate/100)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
        loss_function = nn.CrossEntropyLoss()
        # setup a place to log training metrics
        self.log = {"training_loss":[],
                    "validation_loss": [],
                    "training_accuracy": [],
                    "validation_accuracy": []}
        # now start the training
        for epoch in range(num_epochs):
            
            #training set
            self.train()
            for data, labels in dataloader_dictionary["training_dataloader"]:
                optimizer.zero_grad()
                data = CNN_V1._try_move_to_accelerator(data)
                labels = CNN_V1._try_move_to_accelerator(labels)
                output = self.forward(data) # Size([64, 10, 200]) -> Size([64, 11])
                loss = loss_function(output, labels) # labels.shape == Size([64])
                loss.backward()
                optimizer.step()
                acc = (torch.argmax(output, dim=1) == labels).float().mean()

                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc.item())]
            
            # validation set
            self.eval()
            for data, labels in dataloader_dictionary["validation_dataloader"]:
                data = CNN_V1._try_move_to_accelerator(data)
                labels = CNN_V1._try_move_to_accelerator(labels)
                output = self.forward(data)
                loss = loss_function(output, labels)
                acc = (torch.argmax(output, dim=1) == labels).float().mean()
                # log it
                self.log["validation_loss"] += [(epoch, loss.item())]
                self.log["validation_accuracy"] += [(epoch, acc.item())]
            
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in self.log['training_accuracy'] if i[0]==epoch])
                epoch_valoss = np.mean([i[1] for i in self.log['validation_loss'] if i[0]==epoch])
                epoch_vaacc  = np.mean([i[1] for i in self.log['validation_accuracy'] if i[0]==epoch])

                training_log_callback(epoch, epoch_trloss, epoch_tracc, epoch_valoss, epoch_vaacc)

            scheduler.step() # for CosineAnnealingLR
            # scheduler.step(epoch_valoss) # for ReduceLROnPlateau
            print("Current LR:", scheduler.get_last_lr())

            model_checkpoint.save_if_better(epoch_valoss, self)

            early_stopping(epoch_valoss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.eval()

    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = CNN_V1._try_move_to_accelerator(x)
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = CNN_V1._try_move_to_accelerator(x)
        y = self.forward(x)
        y = self.softmax(y)
        return y.cpu().detach().numpy()
    
    def apply_transfer_strategy(self, strategy):
        """
        strategy: 
            - "finetune_with_fc_reset" (don't freeze any layers, reset FC output layer)
            - "finetune_without_fc_reset" (don't freeze any layers, no reset)
            - "feature_extractor_with_fc_reset" (freeze convolutional layers, reset FC output layer) 
            - "feature_extractor_without_fc_reset" (freeze convolutional layers, no reset)
        """
        if strategy == "finetune_with_fc_reset":
            for parameter in self.parameters():
                parameter.requires_grad = True
            self.output_layer = nn.Linear(self.output_layer.in_features, self.n_output) #  Kaiming Uniform for weights, zeros for bias
            CNN_V1.initialize_with_glorot_weight_zero_bias(self.output_layer, self.generator) # Glorot uniform/xavier for weights, zero for bias
            self.output_layer = CNN_V1._try_move_to_accelerator(self.output_layer)
        
        elif strategy == "finetune_without_fc_reset":
            for parameter in self.parameters():
                parameter.requires_grad = True

        elif strategy == "feature_extractor_with_fc_reset":
            for parameter in self.convolutional_layers.parameters():
                parameter.requires_grad = False
            self.output_layer = nn.Linear(self.output_layer.in_features, self.n_output) #  Kaiming Uniform for weights, zeros for bias
            CNN_V1.initialize_with_glorot_weight_zero_bias(self.output_layer, self.generator) # Glorot uniform/xavier for weights, zero for bias
            self.output_layer = CNN_V1._try_move_to_accelerator(self.output_layer)

        elif strategy == "feature_extractor_without_fc_reset":
            for parameter in self.convolutional_layers.parameters():
                parameter.requires_grad = False

        else:
            raise ValueError("Invalid strategy")

        return self