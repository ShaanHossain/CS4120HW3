import sys
import unittest
import numpy as np
from textclassify_model import TextClassify, f1, generate_tuples_from_file, precision, recall

class TestMiniTraining(unittest.TestCase):

    def run_tests(self):
        unittest.main()

    # def set_lists(self, training, testing):
    #     self.training = training
    #     self.testing = testing

    # training = generate_tuples_from_file(sys.argv[1])
    # testing = generate_tuples_from_file(sys.argv[2])

    # def test_createunigrammodelnolaplace(self):
    #     textclassify_model_baseline = TextClassify()
    #     # self.assertEqual(1, 1, msg="tests constructor for 1, False")

    #check if I need to handle 0 cases
    def test_precision(self):
        ten_zeros = np.zeros(10).tolist()
        ten_ones = np.ones(10).tolist()
        gold_labels = ten_zeros + ten_ones
        predicted_labels = ten_ones + ten_ones

        #10 / 10 + 10
        self.assertEqual(.5, precision(gold_labels, predicted_labels))

        gold_labels = ten_ones + ten_ones

        #20 / 20 + 0
        self.assertEqual(1, precision(gold_labels, predicted_labels))

        gold_labels = ten_zeros + ten_zeros

        #0 / 0 + 20
        self.assertEqual(0, precision(gold_labels, predicted_labels))

    
    def test_recall(self):
        ten_zeros = np.zeros(10).tolist()
        ten_ones = np.ones(10).tolist()
        gold_labels = ten_ones + ten_ones
        predicted_labels = ten_zeros + ten_ones

        #10 / 10 + 10
        self.assertEqual(.5, recall(gold_labels, predicted_labels))

        predicted_labels = ten_zeros + ten_zeros

        #0 / 0 + 20
        self.assertEqual(0, recall(gold_labels, predicted_labels))

        predicted_labels = ten_ones + ten_ones

        #20 / 20 + 0
        self.assertEqual(1, recall(gold_labels, predicted_labels))


    def test_f1(self):
        ten_zeros = np.zeros(10).tolist()
        ten_ones = np.ones(10).tolist()
        gold_labels = ten_ones + ten_ones + ten_zeros
        predicted_labels = ten_zeros + ten_ones + ten_ones

        #2 * .5 * .5 / .5 + .5
        self.assertEqual(.5, f1(gold_labels, predicted_labels))

        gold_labels = ten_ones + ten_ones + ten_ones

        #2 * 1 * (2/3) / (2/3) + 1
        self.assertAlmostEqual(.8, f1(gold_labels, predicted_labels), places=4)

        gold_labels = ten_zeros + ten_ones + ten_zeros

        #2 * .5 * 1 / .5 + 1
        self.assertAlmostEqual((2/3), f1(gold_labels, predicted_labels), places=4)

        gold_labels = ten_zeros + ten_ones + ten_ones

        #2 * 1 * 1 / 1 * 1
        self.assertEqual(1, f1(gold_labels, predicted_labels))

    def test_textclassify_baseline(self):
        model = TextClassify()

        # print(model.featurize(self.training))

if __name__ == "__main__":
  unittest.main()