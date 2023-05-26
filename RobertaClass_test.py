import torch
from RobertaClass import RobertaClass


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = RobertaClass().to(device)
    print(model)
