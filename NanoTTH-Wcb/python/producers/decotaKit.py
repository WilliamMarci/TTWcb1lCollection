import time
import functools
import psutil
import os

global_use=True
# decorator
def timeit(func):
    """
    Timing decorator to measure execution time and memory usage of functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not global_use:
            return func(*args, **kwargs)
        start = time.time()
        print(f"\r[DEBUG] Starting '{func.__name__}'...", end='\r', flush=True)
        result = func(*args, **kwargs)
        end = time.time()
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[DEBUG] Function '{func.__name__}' executed in {end - start:.4f} seconds | Memory usage: {mem_mb:.2f} MB")
        return result
    return wrapper

def getentries(anafunc):
    """
    Get number of entries from an analysis function.
    """
    @functools.wraps(anafunc)
    def wrapper(*args, **kwargs):
        if not global_use:
            return anafunc(*args, **kwargs)
        event = args[1]
        entries = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        print(f"[DEBUG] Total entries to process: {entries}")
        del event
        result = anafunc(*args, **kwargs)
        return result
    
    return wrapper