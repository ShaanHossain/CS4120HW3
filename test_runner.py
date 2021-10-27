import sys
from testing import TestMiniTraining
from textclassify_model import generate_tuples_from_file

def main():

    training = sys.argv[1]
    testing = sys.argv[2]

    tester = TestMiniTraining()
    tester.set_lists(generate_tuples_from_file(training), generate_tuples_from_file(testing))
    tester.run_tests()

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()