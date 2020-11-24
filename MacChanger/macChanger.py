#!/usr/bin/python

"""
Small program that changes the provided interface mac address.
More info about MAC addresses: 
https://en.wikipedia.org/wiki/MAC_address
"""

# TODO: - example usage
#       - verbosity argument
#       - make arguments manadatory/optional

import subprocess
import os
import argparse
import re

# Regex in order to validate a mac address
mac_regex = r"([0-9a-f][0-9a-f]:){5}[0-9a-f][0-9a-f]"


def get_arguments():
    """ This function parses the given arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface",
                        dest="interface",
                        help="Interface to change its MAC address."
                        )
    parser.add_argument("-m", "--mac_address",
                        dest="new_mac",
                        help="New MAC adress."
                        )
    args = parser.parse_args()
    if not args.interface:
        parser.error("[-] Please specify an interface, \
                     use --help for more info.")
    elif not args.new_mac:
        parser.error("[-] Please specify a MAC address, \
                     use --help for more info.")
    return args


def change_mac(interface, new_mac):
    """ This function changes the interface to the new mac. """
    print("[+] Attempting to change MAC adress for " +
          interface + " to " + new_mac)

    subprocess.call(["ifconfig", interface, "down"])
    subprocess.call(["ifconfig", interface, "hw", "ether", new_mac])
    subprocess.call(["ifconfig", interface, "up"])


def get_current_mac(interface):
    """ This function returns the current mac
        for selected interface. """
    interfaceCheck = subprocess.check_output(["ifconfig", interface])\
        .decode('utf-8')
    mac_re = re.search(mac_regex, interfaceCheck)
    if mac_re:
        return mac_re.group(0)
    else:
        print("[-] Could not read MAC address")


if __name__ == "__main__":
    if os.getuid() == 0:
        # Get arguments.
        args = get_arguments()
        print("arguments:" + args.interface + " " + args.new_mac)
        # Get Current MAC address.
        current_mac = get_current_mac(args.interface)
        print("current mac:" + str(current_mac))
        # Check if the address is the same.
        if current_mac == args.new_mac:
            print("MAC address for " + args.interface +
                  " is already " + current_mac)
        else:
            # Change the MAC address.
            print("Current MAC for interface " + args.interface +
                  " : " + str(current_mac))
            change_mac(args.interface, args.new_mac)
            current_mac = get_current_mac(args.interface)
            # Validate if the MAC was changed
            if current_mac == args.new_mac:
                print("[+] MAC address was successfully changed to " +
                      current_mac)
            else:
                print("[-] MAC address did not get changed to " +
                      args.new_mac + ".")
    else:
        print("You need root privileges to run this script")
        print("")
        print("Please try again with sudo")
        print("")
