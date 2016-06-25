
class Config:
	""" For holding model hyperparams """ 
	def __init__(self, batch_size=100, embed_size=50, hidden_size=50, steps=10, max_epochs=10, early_stopping=10, dropout=0.1, lr=0.0001, l2=0.1):
		self.batch_size = batch_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.steps = steps
		self.max_epochs = max_epochs
		self.early_stopping = early_stopping
		self.dropout = dropout
		self.lr = lr
		self.l2 = l2

