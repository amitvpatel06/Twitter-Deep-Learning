import numpy as np
from parse_data import *


def create_data_set(filename, wv_filename, steps=10):
	vocab = Vocab()
	training_set = parse_data_set(vocab, e)
	embedding_data_frame = create_embedding_matrix(vocab, wv_filename)
	return training_set, np.array(embedding_data_frame['word_vectors'])

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)