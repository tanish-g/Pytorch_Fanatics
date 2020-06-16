# Pytorch_Fanatics

Pytorch_Fanatics is a Python library for ease of Computer Vision tasks.This contains a bunch of various tools which will help to create customized codes to train CV models.

###This library includes:
* Dataset class
* LRFinder
* EarlyStopping
* Trainer
* Logger

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pytorch_fanatics.

```bash
pip install pytorch_fanatics
```

## Usage
1)Dataset Class
```python
from pytorch_fanatics.dataloader import Cloader

Cloader(image_path,targets,resize,transforms) # returns {'image':tensor_image,'target':tensor_labels}
#Use it to load images and labels
```

2)LRFinder
```python
from pytorch_fanatics.utils import LRFinder

lr_finder=LRFinder(model,train_dataloader,optimizer,device,initial_lr,final_lr,beta) #Create a object
lr_finder.find()       #To find the lr
lr_finder.plot()       #To plot the graph (Loss V/S lr)
#NOTE:
#Use this only after training the model with all layers (except the last) freezed.
```

3)EarlyStopping
```python
from pytorch_fanatics.utils import EarlyStop

es=EarlyStop(patience=7, mode="max", delta=0.0001) #Create a object
es(epoch_score, model, path)
if es.early_stop=True:
	break
es.reset() # to reset
```

4)Trainer
```python
from pytorch_fanatics.trainer import Trainer

Trainer.train(model,data_loader,optimizer,device) # trains the model

score=Trainer.evaluate(model,data_loader,device,scheduler=None,metric=metrics.accuracy_score,plot=True)
#Use the score to feed for earlystop if used
#plot=True specifies live plot b/w (training and validation) vs num_epochs

Trainer.predict(model,data_loader,device) # returns probability of classes

Trainer.get_log() #returns a DataFrame object of logs 

```

5)Logger
```python
from pytorch_fanatics.logger import Logger

Logger.save(model,optimizer,scheduler,path)  # saves model,optimizer and schedulers

#To load:
checkpoint=Logger.load(path)  #returns a dictionary
model,optimizer,scheduler=checkpoint['model'],checkpoint['optimizer'],checkpoint['scheduler']

#Helps keep track of training.It will restart from where it had stopped.
```

---
**NOTE(Regarding model)**

```python
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.base_model =timm.create_model('resnet18',pretrained=True,num_classes=1)
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss
#Since every loss function has its own format of inputs,To generalise I have created this model.Use this model(edit if required) if you are #using Trainer/LRFinder.For others your simple model will also work fine..

```
---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://github.com/MiHarsh/pytorch_fanatics/blob/master/LICENSE.txt)

