# torch中训练的一般步骤

帮助理解centralized_trainer中的train_impl的笔记

```python
import os
import torch
args.gpu_id="2,7" ; #指定gpu id
args.cuda = not args.no_cuda and torch.cuda.is_available() #作为是否使用cpu的判定
#配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id #这里的赋值必须是字符串，list会报错
device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2
#device_ids=[0,1] 这里的0 就是上述指定 2，是主gpu,  1就是7,模型和数据由主gpu分发
 
if arg.cuda:
    model=model.cuda()  #这里将模型复制到gpu ,默认是cuda('0')，即转到第一个GPU 2
if len(device_id)>1:
    model=torch.nn.DaraParallel(model);#前提是model已经.cuda() 了
 
#前向传播时数据也要cuda(),即复制到主gpu里
for batch_idx, (data, label) in pbar:   
    if args.cuda:
        data,label= data.cuda(),label.cuda();
    data_v = Variable(data)
    target_var = Variable(label)
    prediction= model(data_v,target_var,args)
    #这里的prediction 预测结果是由两个gpu合并过的，并行计算只存在在前向传播里
    #前向传播每个gpu计算量为 batch_size/len(device_ids),等前向传播完了将结果和到主gpu里
    #prediction length=batch_size
 
    criterion = nn.CrossEntropyLoss()
    loss = criterion(prediction,target_var) #计算loss
    optimizer.zero_grad()
    loss.backward()  
    optimizer.step()
```