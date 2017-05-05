import sys
import os
import time

def main():
    idx = 0
    while os.path.exists('img_%03d.png' % idx):
        idx += 1

    while True:
        time.sleep(0.5)
        if os.path.exists('aaa.png'):
            time.sleep(0.5)
            os.rename('aaa.png', 'img_%03d.png' % idx)
            idx += 1

if __name__ == '__main__':
    main()