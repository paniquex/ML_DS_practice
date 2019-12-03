import numpy as np


class RleSequence:
    def __init__(self, input_sequence):
        self.elements, self.amounts = self.encode_rle(input_sequence)
    
    def encode_rle(self, x):
        if len(x) > 0:
            idx_with_differences = np.append(np.where(np.diff(x) != 0)[0],
                                             np.argwhere(x == x[-1])[-1])
            return (x[idx_with_differences], np.diff(np.insert(idx_with_differences, 0, -1)))
        else:
            return None
    
    def decode_rle(self):
        decoded_sequence = np.zeros(np.sum(self.amounts), dtype=self.elements.dtype)
        indx = 0
        for i, element in enumerate(self.elements):
            for tmp in range(self.amounts[i]):
                decoded_sequence[indx] = element
                indx += 1
        return decoded_sequence
    
    def __iter__(self):
        self.curr_idx = 0
        self.curr_amount = 0
        return self
    
    def __next__(self):
        if self.curr_idx < len(self.elements):
            self.curr_amount += 1
            curr_elem = self.elements[self.curr_idx]
            if self.curr_amount >= self.amounts[self.curr_idx]:
                self.curr_amount = 0
                self.curr_idx += 1
            return curr_elem
        else:
            raise(StopIteration)
        
    def __getitem__(self, indx):
        if isinstance(indx, slice):
            start = indx.start
            stop = indx.stop
            step = indx.step
            if start is None:
                start = 0
            if stop is None:
                stop = np.sum(self.amounts)
            if step is None:
                step = 1
            return self.decode_rle()[start:stop:step]
        else:
            return self.decode_rle()[indx]
    
    def __contains__(self, target_elem):
        return target_elem in self.elements
