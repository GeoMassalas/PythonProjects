#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:54:07 2020

@author: gmassalas
"""

import argparse
from time import sleep



def get_arguments():
    """ Function needed in order to parse the arguments given in the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rule", 
                        dest="rule", type=int, 
                        help="Number of rule used. Valid options --> 0-255")
    parser.add_argument("-w", "--width", 
                        dest="width", type=int, 
                        help="Number of columns, preferably odd number. Valid options --> >15")
    parser.add_argument("-he", "--height", 
                        dest="height", type=int, 
                        help="Number of rows that you want printed. Do not use argument if you want infinite generations.")
    parser.add_argument("-t", "--time", 
                        dest="time", type=int, 
                        help="Time that it takes to print two consecutive lines in ms. Default = 100ms")
    
    args = parser.parse_args()
    
    # Rule Validation
    if not args.rule:
        parser.error("You need to provide a rule (0-255)")
    elif (args.rule > 255) | (args.rule < 0):
        parser.error("Rulse needs to be a integer between 0 and 255.")
        
    # Width Validation
    if not args.width:
        parser.error("You need to provide a width larget than 15 columns")
    elif (args.width < 15):
        parser.error("width needs to be more than 15 columns.")
        
    # Time Validation
    if not args.time:
        args.time = 100
        
    # Height Validation
    if args.height:
        if args.height < 1:
            parser.error("Height needs to be larger than 15 columns.")
            
        
    return create_ruleset(args.rule), args.width, args.height, args.time


def char_to_bin(exp):
    """ Function that takes a set of (#,.) characters and returns it in binary form"""
    new_exp = ""
    for i in range(0,len(exp)):
        if exp[i] == "#":
            new_exp += "1"
        else:
            new_exp += "0"
    return new_exp

def bin_to_char(exp):
    """ Fuction that takes a number in binary form and turn it to (# / .) characters"""
    new_exp = ""
    for i in range(0,len(exp)):
        if exp[i] == "1":
            new_exp += "#"
        else:
            new_exp += " "
    return new_exp

def create_ruleset(num):
    f = list(bin_to_char(bin(num).replace("0b", "").rjust(8,"0")))
    return f

def create_initial_string(length):
    f = " " * length
    f = f[:int(length/2)] + '#' + f[int(length/2)+1:]
    return f



if __name__ == "__main__":
    # Get the arguments 
    ruleset, length, height, rep = get_arguments()
    # Calculate delay
    delay = rep/1000
    f = create_initial_string(length)
    print(f)
    if height:
        # Calculate for given height
        for i in range(height):
            sleep(delay)
            newf = " "
            for i in range(1,len(f)-1):
                newf += ruleset[(7 - int( char_to_bin(f[i-1]) + char_to_bin(f[i]) + char_to_bin(f[i + 1]), 2))%8 ]
            newf += ' '
            print(newf)
            f = newf
    else:
        # Infinite calculations
        while(True):
            sleep(delay)
            newf = ""
            for i in range(0,len(f)):
                newf += ruleset[7 - int( char_to_bin(f[i-1]) + char_to_bin(f[i]) + char_to_bin(f[(i + 1) % length]), 2)]
            print(newf)
            f = newf
    
