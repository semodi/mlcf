import time
class Timer:

    def __init__(self, name='TIME', mode='a'):
        self.name = name
        self.start = time.time()
        self.accum = 0
        self.running = True
        self.mode = mode

    def start_timer(self):
        self.start = time.time()
        self.running = True

    def pause(self):
        if self.running:
            self.accum = time.time() - self.start
            self.running = False
        else:
            raise Exception('Timer not running')

    def stop(self):
        if self.running:
            with open(self.name, self.mode) as timefile:
                timefile.write('{} \n'.format(time.time() - \
                    self.start+ self.accum))
            self.running = False
            self.accum = 0
        else:
            raise Exception('Timer not running')


