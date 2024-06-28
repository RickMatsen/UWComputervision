import torch

def optimizer_selector(optim_name, lr, model):
    if optim_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr) # add weight decay if necessary
    elif optim_name == "SGD":
        pass
    elif optim_name == "AdamW":
        pass
    else:
        raise NotImplementedError("Unknown optimizer")
    

def learning_rate_schedular(**kwargs):
    pass