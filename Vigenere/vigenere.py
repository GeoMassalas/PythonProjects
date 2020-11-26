from random import randint
import string
import argparse
import sys

ALPHABETS = ["ascii_lowercase","ascii_uppercase", "ascii_letters", "printable"]
OPERATIONS = ["encrypt", "decrypt"]

def encrypt(text, key):
    print("Encrypting text with key {}.".format(key))
    new_text = ""
    for index, char in enumerate(text):
        alph_index = ALPHABET.find(char)
        key_index = ALPHABET.find(key[index % len(key)])
        if index == -1:
            new_text += char
        else:
            new_text += ALPHABET[(alph_index + key_index) % len(ALPHABET)]
    return new_text


def decrypt(text, key):
    print("Attempting to decrypt text with key {}.".format(key))
    print("-----------------------------------------")
    new_text = ""
    for index, char in enumerate(text):
        alph_index = ALPHABET.find(char)
        key_index = ALPHABET.find(key[index % len(key)])
        if index == -1:
            new_text += char
        else:
            new_text += ALPHABET[(alph_index - key_index) % len(ALPHABET)]
    return new_text


def get_arguments():
    parser = argparse.ArgumentParser(description="Encrypt or decrypt text with ceasar's cipher.")
    parser.add_argument('-t', '--text', 
            dest="text", type=str, required=True,
            help="Text to be encrypted or decrypted. You can also use .txt files.")
    parser.add_argument('-o', '--operation',
            dest="operation", type=str, required=True,
            help="Type of operation. encrypt, decrypt")
    parser.add_argument('-k','--key', required=True, 
            dest="key", type=str, 
            help="Key for the Vigenere Cipher.")
    parser.add_argument('-a','--alphabet',
            dest='alphabet', type=str,
            help="Alphabet used for the operation. Default: ascii_lowercase. OPTIONS: printable, ascii_uppercase, ascii_letters.")
    args = parser.parse_args()
    if args.text[-4:] == '.txt':
        f = open(args.text, "r")
        args.text = f.read()
    if args.operation.lower() not in OPERATIONS:
        parser.error("Invalid Operation. Choose one of: encrypt, decrypt, break")
    if args.alphabet == None:
        args.alphabet = "printable"
    elif args.alphabet.lower() not in ALPHABETS:
        parser.error("Invalid alphabet. Choose one of: ascii_lowercase, ascii_uppercase, ascii_letters, printable.")
    return args.text, args.operation, args.key, args.alphabet 

def specify_alphabet(alphabet):
    if alphabet == "ascii_lowercase":
        return ' ' + string.ascii_lowercase
    elif alphabet == "ascii_uppercase":
        return ' ' + string.ascii_uppercase
    elif alphabet == "ascii_letters":
        return ' ' + string.ascii_letters
    elif alphabet == "printable":
        return string.printable

def write_to_txt(filename, text):
    print("-------------------------------------------")
    print("Processed text: {}...".format(text[:400]))
    print("-------------------------------------------")
    inp = input("Text is too long. Would you like to print it on a txt? (y/n) ")
    if (inp == 'y') | (inp == 'Y'):
        print("Writing to {}".format(filename))
        f = open(filename,"w+")
        f.write(text)
        f.close()

def check_key_against_alphabet(key):
    for char in key:
        if ALPHABET.find(char) == -1:
            print("Charcter ({}) in key, not found in alphabet, please change key or alphabet.".format(char))
            sys.exit()

if __name__ == "__main__":    
    text, operation, key, alphabet = get_arguments()
    ALPHABET = specify_alphabet(alphabet)
    check_key_against_alphabet(key)
    if operation == "encrypt":
        encrypted = encrypt(text, key)
        if len(encrypted) > 400:
            write_to_txt("encrypted.txt", encrypted)
        else:
            print(encrypted)
    elif operation == "decrypt":
        decrypted = decrypt(text, key)
        if len(decrypted) > 400:
            write_to_txt("decrypted.txt", decrypted)
        else:
            print(decrypted)
