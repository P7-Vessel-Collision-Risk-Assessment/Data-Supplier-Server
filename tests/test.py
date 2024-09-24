import unittest
import pandas as pd
import numpy as np

class TestPrepareData(unittest.TestCase):

    def test_test(self):
        data = pd.read_csv("data/testais.csv")
        print(data.to_numpy())
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
