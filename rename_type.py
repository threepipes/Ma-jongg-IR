import sys
import os
import time

def rename(name, path):
    idx = 0
    for filename in os.listdir(path):
        while os.path.exists(path + '/%s_%03d.png' % (name, idx)):
            idx += 1
        os.rename(path + '/' + filename, path + '/%s_%03d.png' % (name, idx))
        idx += 1


def auto_rename():
    jihai = [
        'chu', 'haku', 'hatsu', 'ton', 'nan', 'sha', 'pei'
    ]
    for name in jihai:
        rename(name, name)

    rename_list = ['m', 's', 'p']
    for c in rename_list:
        for i in range(1, 10):
            path = '%s/%i/' % (c, i)
            idx = 0
            for filename in os.listdir(path):
                while os.path.exists(path + '%s%d_%03d.png' % (c, i, idx)):
                    idx += 1
                os.rename(path + filename, path + '%s%d_%03d.png' % (c, i, idx))
                idx += 1

        path = '%s/5r/' % c
        idx = 0
        for filename in os.listdir(path):
            while os.path.exists(path + '%s5r_%03d.png' % (c, idx)):
                idx += 1
            os.rename(path + filename, path + '%s5r_%03d.png' % (c, idx))
            idx += 1



if __name__ == '__main__':
    if len(sys.argv) > 2:
        path = sys.argv[1]
        name = sys.argv[2]
        rename(name, path)
    else:
        print('Auto rename')
        auto_rename()
