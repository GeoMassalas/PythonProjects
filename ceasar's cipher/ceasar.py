from random import randint
import string
import argparse
import operator
import sys

FREQUENCY_TESTS = [' ','a','A','e','E','o', 'O', 'i', 'I']
ALPHABETS = ["ascii_lowercase","ascii_uppercase", "ascii_letters", "printable"]
OPERATIONS = ["encrypt", "decrypt", "break"]

def encrypt(text, key):
    print("Encrypting text with key {}.".format(key))
    new_text = ""
    for char in text:
        index = ALPHABET.find(char)
        if index == -1:
            new_text += char
        else:
            new_text += ALPHABET[(index + key) % len(ALPHABET)]
    return new_text


def decrypt(text, key):
    print("Attempting to decrypt text with key {}.".format(key))
    print("-----------------------------------------")
    new_text = ""
    for char in text:
        index = ALPHABET.find(char)
        if index == -1:
            new_text += char
        else:
            new_text += ALPHABET[(index - key) % len(ALPHABET)]
    return new_text


def get_arguments():
    parser = argparse.ArgumentParser(description="Encrypt or decrypt text with ceasar's cipher.")
    parser.add_argument('-t', '--text', 
            dest="text", type=str, required=True,
            help="Text to be encrypted or decrypted. You can also use .txt files.")
    parser.add_argument('-o', '--operation',
            dest="operation", type=str, required=True,
            help="Type of operation. encrypt, decrypt, break")
    parser.add_argument('-k','--key',
            dest="key", type=int, 
            help="Number of shifts.")
    parser.add_argument('-a','--alphabet',
            dest='alphabet', type=str,
            help="Alphabet used for the operation. Default: ascii_lowercase. OPTIONS: printable, ascii_uppercase, ascii_letters.")
    parser.add_argument('-m', '--method',
            dest='method', type=str,
            help="method to attempt to crack code. Options: brute_force, frequency_analysis(default)")
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
    if args.key == None:
        args.key = randint(2,100)
        print("No key provided, choosing a random key: {}".format(args.key))
    return args.text, args.operation, args.key, args.alphabet, args.method

def specify_alphabet(alphabet):
    if alphabet == "ascii_lowercase":
        return ' ' + string.ascii_lowercase
    elif alphabet == "ascii_uppercase":
        return ' ' + string.ascii_uppercase
    elif alphabet == "ascii_letters":
        return ' ' + string.ascii_letters
    elif alphabet == "printable":
        return string.printable

def crack_brute_force(text):
    print("Attempting all possible keys for alphabet {}.".format(alphabet))
    print("-------------------------------------------------")
    if len(text) > 200:
        text = text[:200]
    for i in range(len(ALPHABET)):
        print("For Key {}, The decrypted text is: {}".format(i, decrypt(text, i)))
        print("-----------------------------------------------")

def crack_frequency_analysis(text):
    letters = {}
    for char in text:
        if char in ALPHABET:
            if char in letters.keys():
                letters[char] += 1
            else:
                letters[char] = 1
    char_index = ALPHABET.find(max(letters.items(), key=operator.itemgetter(1))[0])
    for c in FREQUENCY_TESTS:
        decrypted = decrypt(text,(char_index - ALPHABET.find(c)) % len(ALPHABET))
        if(len(decrypted) > 400):
            print(decrypted[:400] + "...")
        else:
            print(decrypted)
        ans = input("Does this make any sense? y/n ")
        if (ans == 'Y') | (ans == 'y'):
            write_to_txt("decrypted.txt", decrypted)
            sys.exit()
    print("Unable to crack with frequency analysis. Try with brute force.")
    sys.exit()

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

if __name__ == "__main__":    
    text, operation, key, alphabet, method = get_arguments()
    ALPHABET = specify_alphabet(alphabet)
    if operation == "encrypt":
        encrypted = encrypt(text, key)
        if len(encrypted) > 400:
            write_to_txt("encrypted.txt", encrypted)
    elif operation == "decrypt":
        decrypted = decrypt(text, key)
        if len(decrypted) > 400:
            write_to_txt("decrypted.txt", decrypted)
    else:
        if method == "brute_force":
            crack_brute_force(text)
        else:
            print("Atempting frequency analysis on given text.")
            print("-----------------------------------------------")
            crack_frequency_analysis(text)

