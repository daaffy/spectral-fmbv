import unittest

import hello

class TestHello(unittest.TestCase):

    def test_say(self):
        hello.say("yes!")



if __name__ == '__main__':
    unittest.main()