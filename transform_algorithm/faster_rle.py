import torch
import numpy as np
import pandas as pd
import time

import multiprocessing
from collections import ChainMap

def original_rLE_encode(img: np.ndarray, order='F', format=True):
    """
    Args:
    img: binary mask image, shape (r, c)
    order: down-then-right , 应该就是转置后再展开
    format: format determines if the order needs to be preformatted (according to submission rules) or not

    Returns:
    runs: list 
    """
    bytes = img.reshape(img.shape[0] * img.shape[1] , order=order)
    runs = [] # list of run lengths
    r = 0 # the current run length
    pos = 1 # count starts form 1 per WK
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0
    
    if format:
        z = ''
        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z
    else:
        return runs

# RLE with Parallel processing
class Consumer(multiprocessing.Process):
    """Consumer for performing a specific task."""
    def __init__(self, task_queue, result_queue):
        """Initialize consumer, it has a task and result queues."""
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """Actual run of the consumer."""
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            # Fetch answer from task
            answer = next_task()
            self.task_queue.task_done()
            # Put into result queue
            self.result_queue.put(answer)
        return

class RleTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, idx, img):
        """Save image to self."""
        self.img = img
        self.idx = idx

    def __call__(self):
        """When object is called, encode."""
        return {self.idx: original_rLE_encode(self.img)}

class FastRle(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, img, idx):
        """Add a task to perform."""
        self._tasks.put(RleTask(img, idx))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        while not self._results.empty():
            singles.append(self._results.get())
        return dict(ChainMap({}, *singles))

# Proposed faster RLE using pure numpy
#even faster RLE encoder
def toRunLength(x, firstDim = 2):
    
    if firstDim == 2:
        x = np.swapaxes(x, 1,2)
    
    x = (x > 0.5).astype(int)
    x = x.reshape((x.shape[0], -1))    
    x = np.pad(x, ((0,0),(1,1)), 'constant')
    
    x = x[:,1:] - x[:,:-1]
    starts = x > 0
    ends = x < 0
    
    rang = np.arange(x.shape[1])
    
    results = []
    
    for image, imStarts, imEnds in zip(x, starts, ends):
        st = rang[imStarts]
        en = rang[imEnds]
        
#         counts = (en-st).astype(str)
#         st = (st+1).astype(str)
        
#         res = np.stack([st,counts], axis=-1).reshape((-1,))
#         res = np.core.defchararray.join(" ", res)

        res = ""
        for s,e in zip(st,en):
            res += str(s+1) + " " + str(e-s) + " "
            
        results.append(res[:-1])
    #print("called")
        
    return results

# Even faster RLE by using the proposed solution in parallel

class FasterTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, array, startIndex):
        """Save array to self."""
        self.array = array
        self.startIndex = startIndex

    def __call__(self):
        """When object is called, encode."""
        return (toRunLength(self.array), self.startIndex)


class FasterRle(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, array, startIndex):
        """Add a task to perform."""
        self._tasks.put(FasterTask(array, startIndex))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        while not self._results.empty():
            singles.append(self._results.get())
            
        resultDic = dict()
        for rles, start in singles:
            #print('start:', start)
            for i,rle in enumerate(rles):
                #print('i:', i)
                resultDic[str(start+i)] = rle
        return resultDic

# !!!!!!!!!!!Example
example_image = np.random.uniform(0, 1, size=(1000, 101, 101)) > 0.5

# Wrap the FastRle class into a method so we measure the time
def original(array):
    results = {}
    for i, arr in enumerate(array):
        results['%d' % i] = original_rLE_encode(arr)
    return results

def faster(array):
    rle = FastRle(4)
    for i, arr in enumerate(array):
        rle.add('%d' % i, arr)
    return rle.get_results()

def pureNumpy(array):
    rle = toRunLength(array)
    rle = {'%d' % i: row for i,row in enumerate(rle)}
    return rle

def evenFaster(array):
    #make sure you treat this properly when len(array) % 4 != 0
    rle = FasterRle(4)
    subSize = len(array)//4  
    
    for i in range(0,len(array),subSize):
        rle.add(array[i:i+subSize], i)
    return rle.get_results()

def main():
    print("Measuring times: \n")

    print("Original:")
    start_time = time.time()
    original(example_image)
    end_time = time.time()
    print('original time: ', end_time - start_time)

    start_time = time.time()
    print("\nParallel:")
    faster(example_image)
    end_time = time.time()
    print('Parallel time: ', end_time - start_time)

    start_time = time.time()
    print("\nPure numpy:")
    pureNumpy(example_image)
    end_time = time.time()
    print('Pure numpy time: ', end_time - start_time)

    start_time = time.time()
    print("\nEven faster:")
    evenFaster(example_image)
    end_time = time.time()
    print('Even faster time: ', end_time - start_time)

if __name__ == '__main__':
    main()