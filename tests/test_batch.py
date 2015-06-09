import unittest
print __name__

import os
import sys
from roof.MyBatchIterator import MyBatchIterator
import load


class MyBatchTest(unittest.TestCase):
    def setUp(self):
        #X, y = load.load()
        pass

    def testOne(self):
        self.failUnless(True)
    
    def tearDown(self):
        pass

def main():
    unittest.main()

if __name__ == '__main__':
    main()



