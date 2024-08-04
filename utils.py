import torch
import os

def add_model_noise(model, std=0.0001, gpu=True):
    '''
    Add variational noise to models weights: https://ieeexplore.ieee.org/abstract/document/548170
    STD may need some fine tuning...
    '''
    with torch.no_grad():
        for param in model.parameters():
            if gpu:
              param.add_(torch.randn(param.size()).cuda() * std)
            else:
              param.add_(torch.randn(param.size()).cuda() * std)

def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path):
    ''' Load models checkpoint '''
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path)
    scheduler.n_steps = checkpoint['scheduler_n_steps']
    scheduler.multiplier = checkpoint['scheduler_multiplier']
    scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['valid_loss']

def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path):
    ''' Save models checkpoint '''
    torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'scheduler_n_steps': scheduler.n_steps,
            'scheduler_multiplier': scheduler.multiplier,
            'scheduler_warmup_steps': scheduler.warmup_steps,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

def model_size(model, name):
    '''
    Calculate and print models size in num_params and MB

    Parameters:
        model (torch.nn.Module)
        name (str)

    Note:
        - Assume that the models's parameters and buffers are all on the same device (CPU/GPU).
        - The memory size is reported in MB, and the number of parameters is reported in millions.
    '''
    # parameters and the parameter size
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    # models buffer size
    for buffer in model.buffers():
        num_params += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')

def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split('\t')[0]
                 for entry in dictionary]
    # print(char_list)
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')
    return char_list, sos_id, eos_id