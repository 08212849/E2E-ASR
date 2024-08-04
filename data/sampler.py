import random

class BatchSampler(object):
  ''' 
    A custom batch sampler that samples contiguous, sorted indices from the dataset.
    Leads to less padding and faster training. 
  '''
  def __init__(self, sorted_idxs, batch_size):
    self.sorted_idxs = sorted_idxs
    self.batch_size = batch_size
  
  def __iter__(self):
    idxs = self.sorted_idxs.copy()
    while len(idxs):
      batch_num = min(self.batch_size, len(idxs))
      start_idx = random.randint(0, len(idxs) - batch_num)
      batch_idxs = idxs[start_idx:start_idx + batch_num]
      del idxs[start_idx:start_idx + batch_num]
      yield batch_idxs
