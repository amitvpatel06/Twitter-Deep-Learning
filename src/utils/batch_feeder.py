import numpy as np
from parse_data import *

def main():
	vocab = Vocab()	
	x, y, z = create_data_set(vocab, "../data/train.csv", "Test_vecs.txt", steps=10)
	import pdb
	pdb.set_trace()

def create_data_set(vocab, filename, steps=10):
	training_set = parse_data_set(vocab, filename)
	return (training_set['encoded'], 
		training_set['labels'])

def data_iterator(orig_X, orig_y, batch_size=100, label_size=2, shuffle=True):
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
  total_steps = int(np.ceil(len(data_X) / float(batch_size))) - 1
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    y = data_y[batch_start:batch_start + batch_size]
    yield x, y
 

if __name__ == "__main__":
	main()