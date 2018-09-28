#!/bin/bash 
python test_net.py --dataset adas --net res101 --checksession 1 --checkepoch 9 --checkpoint 4999 --cuda
