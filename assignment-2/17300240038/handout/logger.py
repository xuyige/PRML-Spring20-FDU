from time import clock as my_clock

class Logger:
	'''自动日志。

	参数：
	mode：一个字符串列表，字符串代表输出方向。目前有三种可选的方向：
		write：向文件输出
		print：向标准输出输出
		fitlog：向fitlog.add_line输出
		三种方向不冲突
	log_path：日志文件的位置。如果选了"write"的话必须填此项。如果打开了文件最后一定要调用close()

	add_time：是否要在每行末尾输出当前时间

	方法：
	log：输出一个字符串
	close：关闭文件

	'''
	def __init__(self , mode = ["print"] , log_path = None , add_time = True):
		if log_path:
			self.log_fil = open(log_path , "w" , encoding = "utf-8")
		else:
			self.log_fil = None

		self.fitlog = "fitlog" in mode
		self.write = "write" in mode
		self.print = "print" in mode

		if self.write and not log_path:
			raise Exception("Should have a log_path")

		self.add_time = add_time

	def close(self):
		if self.log_fil:
			self.log_fil.close()

	def log(self , content = ""):

		content = self.post_process(content)

		if self.write:
			self.log_fil.write(content + "\n")
			self.log_fil.flush()
		if self.print:
			print (content)
		if self.fitlog:
			import fitlog
			fitlog.add_to_line(content)


	def post_process(self , content):
		if self.add_time:
			content = content + "    |" + " %.2fs" % (my_clock())
		return content