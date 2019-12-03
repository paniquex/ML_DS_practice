import numpy as np


def BatchGenerator(list_of_sequences, batch_size, shuffle=False):
    array_of_sequences = np.array(list_of_sequences, dtype='O')
    if shuffle:
        np.random.shuffle(array_of_sequences.T)
    split_size = array_of_sequences.shape[1] // batch_size \
                                + (array_of_sequences.shape[1] % batch_size)
    for i in range(0, array_of_sequences.shape[1], batch_size):
        yield array_of_sequences[:, i:(i + batch_size)].tolist()
