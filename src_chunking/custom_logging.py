class Logger:
    """
    info <--> debug
    console_debug <--> error
    """
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(log_file, 'a', encoding='utf-8')
        return
    
    def log(self, message):
        self.file.write(message)
        self.flush()
        return

    def print(self, *args, end='\n'):
        for arg in args:
            print(arg, end=' ')
        print(end=end)
        return

    def debug(self, *args):
        message = ''
        for arg in args:
            message += f'{arg}' + ' '
        message = message + '\n'
        self.log(message)
        return
    
    def info(self, *args):
        message = ''
        for arg in args:
            message += f'{arg}' + ' '
        message = message + '\n'
        self.log(message)
        return
    
    def console_debug(self, *args):
        self.info(*args)
        self.print(*args)
        return
    
    def error(self, *args):
        self.info(*args)
        self.print(*args)
        return
    
    def flush(self):
        self.file.close()
        self.file = open(self.log_file, 'a', encoding='utf-8')
        return