import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt

class LRFinder:
    def __init__(self,model,train_dataloader,optimizer,device,initial_lr=1e-8,final_lr=10,beta=0.98):
        self.model=model
        self.train_dataloader=train_dataloader
        self.optimizer=optimizer
        self.beta=beta
        self.initial_lr=initial_lr
        self.final_lr=final_lr
        self.device=device
        self.optimizer.param_groups[0]['lr'] = self.initial_lr
        self.num = len(self.train_dataloader)-1
        self.mult = (self.final_lr / self.initial_lr) ** (1/self.num)
        self.loss_list=[]
        self.log_lrs=[]
    def find(self):
        best_loss=np.inf
        avg_loss=0.0
        train=self.train_dataloader
        tk = tqdm(train, total=len(train), position=0, leave=True)
        smoothed_loss=0.0
        for i,data in enumerate(tk):
            for key, value in data.items():
                data[key] = Variable(value.to(self.device))
            self.model.train()
            self.optimizer.zero_grad()
            _, loss = self.model(**data)
            avg_loss = self.beta * avg_loss + (1-self.beta) *loss.data.item()
            #Compute the smoothed loss
            smoothed_loss += avg_loss / (1 - self.beta**(i+1))   
            #Stop if the loss is exploding
            if i > 0 and smoothed_loss > 1000 * best_loss:
                break
            #Record the best loss
            if smoothed_loss < best_loss or i==0:
                best_loss = smoothed_loss
            #Store the values
            self.loss_list.append(smoothed_loss/len(train.dataset))
            self.log_lrs.append(np.log10(self.initial_lr))
            loss.backward()
            self.optimizer.step()
            #Update the lr for the next step
            self.initial_lr *= self.mult
            self.optimizer.param_groups[0]['lr'] = self.initial_lr
            
        
                
    def plot(self):
        plt.plot( self.log_lrs, self.loss_list,color='blue')
        plt.title('Plot b/w losses and lrs')
        plt.xlabel('LRs----> 1e')
        plt.ylabel('Loss----->')
        plt.show()

