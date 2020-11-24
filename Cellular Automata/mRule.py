#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:24:54 2020

@author: gmassalas

Usage: python3 mRule.py -w $COLUMNS -he $LINES
"""

import argparse
from time import sleep
import random



def get_arguments():
    """ Function needed in order to parse the arguments given in the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", 
                        dest="width", type=int, 
                        help="Number of columns, preferably odd number. Valid options --> >15")
    parser.add_argument("-he", "--height", 
                        dest="height", type=int, 
                        help="Number of rows ")
    parser.add_argument("-t", "--time", 
                        dest="time", type=int, 
                        help="Time that it takes to move from one state to an other in ms. Default = 100ms")
    
    args = parser.parse_args()
    
        
    # Width Validation
    if not args.width:
        parser.error("You need to provide a width larger than 15 columns")
    elif (args.width < 15):
        parser.error("width needs to be more than 15 columns.")
        
    # Time Validation
    if not args.time:
        args.time = 100
        
    # Height Validation
    if args.height:
        if args.height < 1:
            parser.error("Height needs to be larger than 15 columns.")
            
        
    return  args.width, args.height, args.time

def print_state(CA):
    f = ""
    for i in range(height): 
        f += "\n"
        for j in range(length):
            if CA[i][j] == 1:
                f += "#"
            else:
                f += " "
    print(f)
    
def update(height, length, CA):
    nextCA = [[0 for x in range(length)] for y in range(height)]
    for i in range(height):
        for j in range(length):
            lives = 0
            
            if CA[(i)%height][(j+1)%length] == 1:
                lives += 1
            if CA[(i)%height][(j-1)%length] == 1:
                lives += 1
            if CA[(i+1)%height][(j)%length] == 1:
                lives += 1
            if CA[(i-1)%height][(j)%length] == 1:
                lives += 1
            if CA[(i+1)%height][(j+1)%length] == 1:
                lives += 1
            if CA[(i+1)%height][(j-1)%length] == 1:
                lives += 1
            if CA[(i-1)%height][(j+1)%length] == 1:
                lives += 1
            if CA[(i-1)%height][(j-1)%length] == 1:
                lives += 1
            if CA[(i)%height][(j)%length] == 1:
                lives += 1

            if lives >= 5:
                nextCA[i][j] = 1
            else:
                nextCA[i][j] = CA[i][j]

    return nextCA
    
if __name__ == "__main__":
    # Get the arguments 
    length, height, rep = get_arguments()
    height -= 1
    # Calculate delay
    delay = rep/1000
    # Initialize board
    CA = [[0 for x in range(length)] for y in range(height)]
    # Random seed the board
    for i in range(height):
        for j in range(length):
            if random.randint(1,100) < 33:
                CA[i][j] = 1
            else:
                CA[i][j] = 0
    print_state(CA)
    
    while(True):
        print_state(CA)
        CA = update(height, length, CA)
        sleep(delay)
        

        
    
