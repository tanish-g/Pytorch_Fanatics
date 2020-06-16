import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation
from matplotlib import style
plt.style.use('dark_background')
import sys
sys.tracebacklimit = 0

log=[]
train_list=[]
val_list=[]
class Trainer:
    @staticmethod
    def __init__():
        global epoch,val_loss,train_loss,log,train_list,val_list,lr
        log=[]
        train_list=[]
        val_list=[]

    def reset():
        global epoch,val_loss,train_loss,log,train_list,val_list
        epoch=0
        val_loss=0.0
        train_loss=0.0
        log=[]
        train_list=[]
        val_list=[]


    @staticmethod
    def train(model,data_loader,optimizer,device):
        global train_loss,lr,train_list
        model.train()
        optimizer.zero_grad()
        train_loss=0.0
        lr=optimizer.param_groups[0]['lr']
        tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
        for idx,data in enumerate(tk):
            for key, value in data.items():
                data[key] = Variable(value.to(device))
            outputs, loss = model(**data)
            _,preds=torch.max(outputs.data,1)
            loss.backward()
            optimizer.step()
            train_loss+=float((loss.data).item())/len(data['targets'].data)
            tk.set_postfix(loss=train_loss)
        train_list.append(train_loss)

    @staticmethod
    def evaluate(model,data_loader,device,scheduler=None,metric=metrics.accuracy_score,plot=True):
        global epoch,val_loss,train_loss,log,train_list,val_list,lr
        final_predictions=[]
        pred_proba=[]
        targets=[]
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
            for idx,data in enumerate(tk):
                for key, value in data.items():
                    data[key] = Variable(value.to(device))
                outputs, loss = model(**data)
                _,preds=torch.max(outputs.data,1)
                val_loss+=float((loss.data).item())/len(data['targets'].data)
                final_predictions.append(preds.cpu())
                pred_proba.append(outputs.data.cpu)
                targets.append(data['targets'].data.cpu())
        val_list.append(val_loss)
        pred_proba=np.vstack(pred_proba)
        pred = np.hstack((final_predictions))
        targets=np.hstack(targets)
        if metric==metrics.f1_score:
            metric_score=metric(y_true=targets,y_pred=pred,average='micro') #Used micro as this is better option for imbalanced datas.
            if scheduler is not None:
                scheduler.step(metric_score)
            print(f'--------Validation f1_score:{metric_score}')
        else:
            metric_score=metric(targets,pred)
            if scheduler is not None:
                scheduler.step(metric_score)
            print(f'-------Validation metric_score:{metric_score}')
        log.append([train_loss,val_loss,metric_score,lr])
        clear_output(wait=True)
        print(pd.DataFrame(log,columns=['Train_Loss','Valid_Loss','Metric_Score','Current_LR']),flush=True)
        if plot:
            Trainer.Plotit()
        return metric_score
    
    @staticmethod
    def predict(model,data_loader,device):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
            for idx,data in enumerate(tk):
                for key, value in data.items():
                    data[key] = Variable(value.to(device))
                outputs, loss = model(**data)
                final_predictions.append(outputs.data.cpu())
        pred=np.vstack(final_predictions)
        return pred
    
    def get_log():
        return pd.DataFrame(log,columns=['Train_Loss','Valid_Loss','Metric_Score','Current_LR'])

    def animate():
        plt.plot(np.arange(len(train_list)),train_list,color='blue' )
        plt.plot(np.arange(len(val_list)),val_list,color='orange' )
        plt.xlabel('Epochs------>')
        plt.ylabel('Losses------->')
        plt.title('Train and Val Loss Analysis')
        plt.legend(['Train_loss','Val_loss'])
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    def Plotit():
        ani = FuncAnimation(plt.gcf(), Trainer.animate(), 5000)
        plt.tight_layout()
        plt.show()
        




