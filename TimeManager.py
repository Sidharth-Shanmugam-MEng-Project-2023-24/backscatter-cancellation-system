import time




class Timer:

    def __init__(self):
        self.start_timestamp = 0




    def stop(self):
        return time.perf_counter() - self.start_timestamp

