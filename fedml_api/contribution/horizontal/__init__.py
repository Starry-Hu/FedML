import torch
import torch.utils.data as Data

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(20, 1, 10)       # y data (torch tensor)

metrics1 = {
    'y_pred': [x,y,x]
}

metrics2 = {
    'y_pred': [y,x,y]
}

if __name__ == '__main__':
    batch_sum = torch.tensor([], device='cpu')

    base_pred = metrics1['y_pred']
    delete_pred = metrics2['y_pred']
    instance_num = 0
    for batch in range(len(base_pred)):
        instance_num += base_pred[batch].size()[0]
        batch_one = torch.sum(torch.abs(base_pred[batch] - delete_pred[batch]))
        batch_sum = torch.cat((batch_sum, torch.tensor([batch_one])), 0)
    result = torch.sum(batch_sum) / instance_num
    print(result)