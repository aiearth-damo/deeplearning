# -*- coding: utf-8 -*-

import fcntl
import time


class Flock:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fd = open(self.file_path, "w")

    def __del__(self):
        self.fd.close()
        
    def lock(self, no_blocking=False):
        if no_blocking:
            flag = fcntl.LOCK_EX | fcntl.LOCK_NB
        else:
            flag = fcntl.LOCK_EX
        fcntl.flock(self.fd, flag)

    def un_lock(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)





if __name__ == "__main__":
    from multiprocessing import Process
    def worker():
        f_path = "test.lock"
        flock = Flock(f_path)
        flock.lock()
        time.sleep(10)
        print("hahahah")
        flock.un_lock()
    Process(target=worker).start()
    Process(target=worker).start() 