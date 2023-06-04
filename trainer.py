import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Callable
from .model.transformer import Transformer
from .loss import TransformerLoss
from .metric import BLEU
from .scheduler import Scheduler
from .model.utils.mask import generate_padding_mask, generate_look_ahead_mask
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

class TransformerTrainer:
    def __init__(self,
                 encoder_token_size: int,
                 decoder_token_size: int,
                 n: int = 6,
                 d_model: int = 512,
                 heads: int = 8,
                 d_ff: int = 2048,
                 dropout_rate: float = 0.1,
                 eps: float = 0.1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 device: str = 'cpu',
                 checkpoint: str = None) -> None:
        
        self.device = device
        self.loss = 0.0
        self.epoch = 0
        
        self.model = Transformer(
            encoder_token_size=encoder_token_size,
            decoder_token_size=decoder_token_size,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            eps=eps,
            activation=activation
        )

        self.optimizer = optim.Adam(params=self.model.parameters(), betas=[0.9, 0.98], eps=1e-9)

        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)
        
        self.loss_function = TransformerLoss()
        self.scheduler = Scheduler(optimizer=self.optimizer, d_model=d_model, wramup_steps=4000)
        self.metric = BLEU() 
        
        self.model.to(self.device)

    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)
    
    def train_step(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, labels: torch.Tensor):
        padding_mask = generate_padding_mask(encoder_inputs)
        look_ahead_mask = generate_look_ahead_mask(decoder_inputs)

        outputs = self.model(encoder_inputs, decoder_inputs, padding_mask, look_ahead_mask)

        self.loss += self.loss_function(outputs, labels)
        self.scheduler.step()

    def build_dataset(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int):
        dataset = TensorDataset(inputs, outputs)
        return DataLoader(dataset=dataset, batch_size=batch_size)
    
    def fit(self, 
            inputs: torch.Tensor, 
            outputs: torch.Tensor, 
            epochs: int, 
            batch_size: int, 
            mini_batch: int, 
            validation_split: float = None,
            validation_data: tuple[torch.Tensor, torch.Tensor] = None, 
            validation_batch: int = None,
            **kwargs) -> None:
        
        validation = False
        if validation_data is not None:
            val_inputs, val_outputs = validation_data
            validation = True
        elif validation_split is not None:
            inputs, val_inputs,outputs, val_outputs = train_test_split(inputs, outputs, test_size=validation_split)
            validation = True

        if validation:
            if validation_batch is None:
                validation_batch = batch_size

        dataloader = self.build_dataset(inputs, outputs, batch_size)
        total_batches = len(dataloader)
        
        loss_epoch = 0.0

        self.model.train()
        for _ in range(epochs):
            count = 0
            for index, data in enumerate(dataloader, 0):
                encoder_inputs = data[0].to(self.device)
                decoder_inputs = data[1][:, :-1].to(self.device)
                labels = data[1][:, 1:].to(self.device)

                self.train_step(encoder_inputs, decoder_inputs, labels)
                count += 1

                if count == mini_batch or count == total_batches-1:
                    print(f"Epoch: {self.epoch+1} Batch {index + 1} Loss: {(self.loss/count):.4f}")
                    loss_epoch += self.loss
                    self.loss = 0.0

            print(f"Epoch: {self.epoch + 1} Train Loss: {(loss_epoch/total_batches):.4f}")

            if validation:
                val_loss, val_score = self.evaluate(val_inputs, val_outputs, validation_batch)
                print(f"Epoch: {self.epoch + 1} Train Loss: {(val_loss):.4f} BLEU Score: {(val_score):.4f}")
            
            self.epoch += 1

    def evaluate(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        dataloader = self.build_dataset(inputs, outputs, batch_size=batch_size)
        num_batches = len(dataloader)
        loss = 0.0
        score = 0.0
        self.model.eval()
        for _, data in enumerate(dataloader):
            encoder_inputs = data[0].to(self.device)
            decoder_inputs = data[1][:, :-1].to(self.device)
            labels = data[1][:, 1:].to(self.device)

            padding_mask = generate_padding_mask(encoder_inputs)
            look_ahead_mask = generate_look_ahead_mask(decoder_inputs)

            outputs = self.model(encoder_inputs, decoder_inputs, padding_mask, look_ahead_mask)
            loss += self.loss_function(outputs, labels)

            _, predicted = torch.maximum(outputs)
            score += self.metric.score(predicted, labels)

        return loss/num_batches, score/num_batches
    
    def predict(self, x: torch.Tensor, start_token: int, end_token: int , max_steps: int = 256):
        decoder_input = torch.tensor([[start_token]])
        encoder_output = self.model.encoder(x, None)
        self.model.eval()
        for _ in range(max_steps):
            output = self.model.decoder(decoder_input, encoder_output, None, None)

            _, predicted = torch.maximum(output)

            if predicted == end_token:
                break

            decoder_input = torch.concat([decoder_input, predicted.unsqueeze(0)], dim=-1)

        return decoder_input

    