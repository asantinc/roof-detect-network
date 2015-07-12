import time
import cProfile

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose=verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end -self.start
        self.msecs = self.secs * 1000
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

    
