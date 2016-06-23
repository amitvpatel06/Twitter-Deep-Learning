
class Config:
	""" For holding model hyperparams """ 
	def __init__(self, batch_size, embed_size, hidden_size, num_steps, max_epochs, early_stopping, dropout, lr, l2):
		self.batch_size = batch_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.num_steps = num_steps
		self.max_epochs = max_epochs
		self.early_stopping = early_stopping
		self.dropout = dropout
		self.lr = lr
		self.l2 = l2

