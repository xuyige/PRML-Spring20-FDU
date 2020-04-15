import time

init_time = time.time()

class Logger:
	def __init__(self , path , add_time = True , print = True):
		self.fil = open(path , "w" , encoding = "utf-8")
		self.add_time = add_time
		self.print = print
	
	def close(self):
		self.fil.close()

	def now_time(self):
		return time.time() - init_time

	def append_time_str(self , x):
		add_pos = 100
		if "\n" in x:
			len_x = len(x.split("\n")[-1])
		else:
			len_x = len(x)

		if len_x < add_pos:
			x += " " * (add_pos - len_x)
		x += "     | time = %.2fs" % (self.now_time())
		return x

	def log(self , x = ""):
		if self.add_time:
			x = self.append_time_str(str(x))
		if self.print:
			print (x)
		self.fil.write(str(x) + "\n")
		self.fil.flush()

	def log_no_print(self , x = ""):
		if self.add_time:
			x = self.append_time_str(str(x))

		self.fil.write(str(x) + "\n")
		self.fil.flush()

	def add_line(self):
		self.log("-----------------------------------------------------------------")
