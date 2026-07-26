import torch
import torch.optim as optim
import optuna
epoch = 100
optimizer = optim.Adam(model.parameters(),lr = 0.005)

#1
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.2)
''' after model train epoch every 10 times,每训练 10 个 epoch，把学习率乘以 0.2
    一般放在 每个 epoch 结束后。
'''
scheduler.step()
print(
    epoch,
    optimizer.param_groups[0]["lr"]
)

#2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'max',
                                                 factor = 0.2 ,patience = 3)
scheduler.step(val_accuracy)
'''
模型性能长期没有提升，就自动降低 learning rate,监控指标越大越好（当然也可以选择越小越好，
主要是看调用的时候选择的监控指标是什么，lr = lr × factor
'''

#3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max = n_epochs,eta_min = 0.002)
scheduler.step()
'''
学习率按照余弦曲线逐渐下降,
CosineAnnealingLR 不是完整使用一个余弦周期，而是只取了余弦函数的一半（半个周期）
T_max表示一个完整版个余弦周期需要多少个 epoch，n_epochs=100，则第一百个epoch，lr = eta_min = 0.002
'''

#4
optimizer2 = optim.Adam(model.parameters(),lr = 0.001 , weight_decay = 1e-5)

'''
    penalizing large weights by adding the squares of the weight to the loss function 
    让权重稍微往0靠近，防止模型参数变得过大，减少过拟合
    Loss（new）​=Loss+λ∑w^2 ，第一部份表示预测要准，loss要小；第二部份标售权重要小
    L2 正则化形式
'''

#5
'''
early stopping: number of epochs to wait if improvement stalls
在验证集上性能不再提高的时候停止训练
'''

#6
'''
for a convolutional layer with 64 chanels,naomalizes the inputs to a layer
nn.BatchNorm2d(64)
'''

#7
'''
超参数自动搜索
训练神经网络时，有很多参数不是模型参数，而是超参数；为了减少人工调
grid search
random search
Optuna:tree-structured parzen estimator:首先随机选择一个超参数组合
Optuna 每调用一次 objective()，
就随机/智能选择一组超参数 → 创建模型 → 训练 → 返回一个评价指标（比如 accuracy/loss
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def objective(trial,device):
    # === choose hyperparameters ===

    #feature extractor parameters
    n_layers = trial.suggest_int("n_layers",1,3)  #Optuna 从1-3选一个整数
    n_filters = [trial.suggest_int(f"n_filters_{i}",16,128) for i in range(n_layers)]
    kernel_size = [trial.suggest_categorical(f"kernel_size_{i}",[3,5]) for i in 
                   range(n_layers)]
    
    dropout_rate = trial.suggest_float("dropout_rate",0.1,0.5)
    fc_size = trial.suggest_int("fc_size",64,256)

def objective(trial,device):

    n_epochs = 10
    helper_utils.train_model(model = model,optimizer = optimizer,
                             train_dataloader = train_loader,n_epochs = n_epochs,loss_fcn = loss_fcn
                             device = device)
    accuracy = helper_utils.evaluate_accuracy(model,val_loader,device)

    return accuracy

study = optuna.create_study(direction = 'maximize')
n_trials = 20
study.optimize(lambda trial:objective(trial,device),n_trials = n_trials)

#8

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb

'''
计算一个 PyTorch 模型占用多少内存（MB）
param.nelement()这个 tensor 里面有多少个元素
'''

#9
import time
def measure_inference_time(model,input_data,num_iterations =100):
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    #warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    if device.type == "cuda":
        torch.cuda.synchronize()
    #begin
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()


    total_time = end_time - start_time
    avg_time = total_time / num_iterations

    return avg_time

#10
def select_best_model_constraint_based(results,max_size_mb,max_inference_ms):
    results = results.to_dict(orient = "index")

    #filter models that meet both size and inference time constraints
    viable_models ={
        name:metrics for name,metrics in results.items()
        if metrics["model_size_mb"] <= max_size_mb and
           metrics["inferenc_time_ms"] <= max_inference_ms
    }
    if not viable_models:
        #if no models statisfy the constraints,inform the user
        print("No models meet all constrains.Consider relaxing constrains.")
        return None
    
    best_model = max(viable_models.items(),key = lambda x:x[1]["accuracy"])
    return best_model[0],viable_models
'''
约束条件下选择最佳模型
viable_models[name] = metrics
python中dictionary使用items()得到是会是（key,value)的二元组tuple
max(viable_models.items(),key = lambda x:x[1]["accuracy"])
x[0] = name,x[1] = value
是在告诉二元组tuple按照value中“accuracy”对应的值选最max的，最后返回值还是tuple
'''

#11
def select_model_weighted(results,weights = None):
    results = results.to_dict(orient = "index")

    if weights is None:
        weights = {"accuracy":0.5,"model_size_mb":0.2,"inference_time_ms":0.3}
    metrics = list(weights.keys())
    normalized = {name:{} for name in results}
    for metric in metrics:
        values = [res[metric] for res in results.values()]
        min_val,max_val = min(values),max(values)
        range_val = max_val - min_val if max_val != min_val else 1.0
        #avoid division by zero

        for name,res in results.items():
            value = res[metric]
            if metric == "accuracy":
                norm_value = (value - min_val)/range_val
            else:
                #for model size and inference time:lower is better->inverse normalization
                norm_value = 1 - (value - min_val)/range_val
            normalized[name][metric] = norm_value
    scores={}
    for name,metrics in normalized.items():
        score=0
        for metric,weight in weights.items():
            score += metrics[metric]*weight
        scores[name]=score

    best_model = max(
    scores,
    key=scores.get
)
    return best_model,scores
'''
accuracy、模型大小、推理速度都有考虑 → 综合打分最高
对参数进行归一化然后加权计算得分，从而比较模型
'''