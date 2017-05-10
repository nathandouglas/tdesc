#!/usr/bin/env python

"""
    runner.py
"""

import os
import sys
import argparse
import numpy as np

import urllib
import cStringIO
from time import time

from multiprocessing import Process, Queue
from Queue import Empty

from workers import VGG16Worker

# --
# Init

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--n-threads', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--crow', action="store_true")
    return parser.parse_args()

# --
# Threaded IO

def prep_images(in_, out_, imread, timeout):
    while True:
        try:
            path = in_.get(timeout=timeout)
            # try:
            img = imread(path)
            out_.put((path, img))
            # except KeyboardInterrupt:
            #     raise
            # except:
            #     print >> sys.stderr, "prep_images: Error at %s" % path
        
        except KeyboardInterrupt:
            raise
        
        except Empty:
            return

def read_stdin(gen, out_):
    for line in gen:
        out_.put(line.strip())

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    if args.model == 'vgg16':
        worker = VGG16Worker(args.crow)
    else:
        raise Exception()
    
    # Thread to read filenames from stdin
    filenames = Queue()
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    stdin_reader = Process(target=read_stdin, args=(newstdin, filenames))
    stdin_reader.start()
    
    # Thread to load images    
    processed_images = Queue()
    image_processors = [Process(target=prep_images, args=(filenames, processed_images, worker.imread, args.timeout)) for _ in range(args.n_threads)]
    for image_processor in image_processors:
        image_processor.start()
    
    i = 0
    start_time = time()
    while True:
        
        try:
            path, img = processed_images.get(timeout=args.timeout)
            worker.featurize(path, img)
            
            i += 1
            if not i % 100:
                print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)
            
        except KeyboardInterrupt:
            raise
        except Empty:
            worker.close()
            os._exit(0)
        except:
            pass
    
    print >> sys.stderr, "%d images | %f seconds " % (i, time() - start_time)