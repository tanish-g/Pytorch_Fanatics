import torch

class Logger:
    #This has been specifically designed to store just the last epoch 
    #    (you can use it to save other epochs too..)
    def save(model,optimizer,scheduler,path):
        checkpoint = { 
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler}
        torch.save(checkpoint, path)
    
    def load(path):
        checkpoint=torch.load(path)
        return checkpoint

