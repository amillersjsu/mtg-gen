from tensorflow.keras.callbacks import Callback
import pickle

class HistoryCheckPoint(Callback):
	def __init__(self,file='history',**kargs):
		super(HistoryCheckPoint,self).__init__(**kargs)
		self.history = {}
		self.history['loss'] = []
		self.history['val_loss'] = []
		self.file = file

	def on_epoch_end(self, epoch, logs={}):
		self.history['loss'].append(logs.get("loss"))
		self.history['val_loss'].append(logs.get("val_loss"))
		pickle.dump(self.history, open(self.file + "_epoch=" + str(epoch+1) + ".p", "wb"))