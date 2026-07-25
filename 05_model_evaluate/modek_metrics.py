import torch
import torchmetrics

'''
accuracy： percentage of correct prediction made by the model
    accuracy = correct prediction / all prediction

precision measures how often the model's positive predictions are correct
    precision = true positive/(true positive + false positive)

recall measures how often the model identifies true positives from all positives
    recall = true positive / (true positives + false negatives)

F1 score balances false positives and false negatives
    F1 score = 2 * (precision * recall) / (precision + recall)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate_metrics(model,val_dataloader,device,num_classes = 10):

    accuracy_metrics = torchmetrics.Accuracy(
        task = "multiclasses",num_classes = num_classes,average = "macro"
    ).to(device)
    
    precission_metrics = torchmetrics.Precision(
        task = "multiclasses",num_classes = num_classes,average = "macro"
    ).to(device)

    recall_metrics = torchmetrics.Recall(
        task = "multiclasses",num_classes = num_classes,average = "macro"
    ).to(device)

    F1score_metrics = torchmetrics.F1Score(
        task = "multiclasses" ,num_classes = num_classes,average = "macro"
    ).to(device)