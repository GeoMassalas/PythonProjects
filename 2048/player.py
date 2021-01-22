from puzzle import puzzle
from pynput.keyboard import Listener, Key
import sys
import threading

def on_release(key):
    if p.is_valid() and key != Key.esc:
        if key == Key.up:
            p.move("up")
        elif key == Key.down:
            p.move("down")
        elif key == Key.left:
            p.move("left")
        elif key == Key.right:
            p.move("right")
        else:
            pass
    else:
        print("Game Over!")
        x.kill()
        sys.exit()

def display():
    while(True):
        p.print_state()

if __name__ == "__main__":
    p = puzzle()
    x = threading.Thread(target=display)
    x.deamon = True
    x.start()
    with Listener(on_release=on_release) as listener:  # Create an instance of Listener
        listener.join()
