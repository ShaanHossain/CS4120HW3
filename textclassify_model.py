# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math

"""
Shaan Hossain
Your name and file comment here:
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""
def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples

def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  
  true_pos = 0 #gold pos sys pos
  false_pos = 0 #gold neg sys pos

  for i in range(0, len(gold_labels)):
    if(gold_labels[i] == predicted_labels[i] == "1"):
      true_pos += 1
    elif(gold_labels[i] == "0" and predicted_labels[i] == "1"):
      false_pos += 1

  if(true_pos == 0):
    return 0
  else:
    return float(true_pos / (true_pos + false_pos))


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  
  true_pos = 0 #gold pos sys pos
  false_neg = 0 #gold pos sys neg

  for i in range(0, len(gold_labels)):
    if(gold_labels[i] == predicted_labels[i] == "1"):
      true_pos += 1
    elif(gold_labels[i] == "1" and predicted_labels[i] == "0"):
      false_neg += 1

  if(true_pos == 0):
    return 0
  else:
    return float(true_pos / (true_pos + false_neg))

def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  
  precision_value = float(precision(gold_labels, predicted_labels))
  recall_value = float(recall(gold_labels, predicted_labels))
  top = float(2 * precision_value * recall_value)
  bottom = precision_value + recall_value

  if(top == 0):
    return 0
  else:
    return float(top / bottom)


"""
Implement any other non-required functions here
"""

def predict_examples(model, examples):
  
  gold_labels = []
  predicted_labels = []
  
  for e in examples: 
    gold_labels.append(e[2])
    predicted_labels.append(model.classify(e[1]))

  return gold_labels, predicted_labels

def report_scores(gold_labels, predicted_labels):
  print("Precision: " + str(precision(gold_labels, predicted_labels)))
  print("Recall: " + str(recall(gold_labels, predicted_labels)))
  print("F1-Score: " + str(f1(gold_labels, predicted_labels)))

"""
implement your TextClassify class here
"""
class TextClassify:


  def __init__(self):
    # do whatever you need to do to set up your class here
    self.vocab = set()
    self.class_zero = Counter()
    self.class_zero_doc_count = 0
    self.class_zero_feature_count = 0
    self.class_one = Counter()
    self.class_one_doc_count = 0
    self.class_one_feature_count = 0
    self.total_docs = 0
    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
     
    for e in examples:

      class_label = None

      features = self.featurize(e[1])

      #determining which class corresponds to 
      if(e[2] == '0'):
        class_label = self.class_zero
        self.class_zero_feature_count += len(features)
        self.class_zero_doc_count += 1
      else:
        class_label = self.class_one
        self.class_one_feature_count += len(features)
        self.class_one_doc_count += 1

      for f in features:
        if(f[1] == True):
          #adding feature to vocabulary
          self.vocab.add(f[0])
          #adding feature to class to keep track of counts
          class_label[f[0]] += 1
          

    self.total_docs = len(examples)
        
  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """

    score_mappings = {
      "0": np.log(self.class_zero_doc_count / self.total_docs),
      "1": np.log(self.class_one_doc_count / self.total_docs)
    }

    features = self.featurize(data)

    for f in features:

      if(f[0] in self.class_zero):
        cond_prob_zero = np.log((self.class_zero[f[0]] + 1) / (self.class_zero_feature_count + len(self.vocab)))
      elif(f[0] in self.vocab):
        cond_prob_zero = np.log(1 / (self.class_zero_feature_count + len(self.vocab)))
      else:
        cond_prob_zero = 0

      if(f[0] in self.class_one):
        cond_prob_one = np.log((self.class_one[f[0]] + 1) / (self.class_one_feature_count + len(self.vocab)))
      elif(f[0] in self.vocab):
        cond_prob_one = np.log(1 / (self.class_one_feature_count + len(self.vocab)))
      else:
        cond_prob_one = 0

      score_mappings["0"] += cond_prob_zero
      score_mappings["1"] += cond_prob_one

    score_mappings["0"] = np.exp(score_mappings["0"])
    score_mappings["1"] = np.exp(score_mappings["1"])

    return score_mappings

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    score_mappings = self.score(data)

    #update this logic to return max or the first thing in sorted list

    # score_mappings["2"] = 0.009015777610818933 

    # print(score_mappings)

    max_value = score_mappings[max(score_mappings, key=score_mappings.get)]

    # print(max_value)

    score_mappings = dict(filter(lambda x: x[1] == max_value, score_mappings.items()))

    # print(score_mappings)

    return sorted(score_mappings)[0]

  def featurize(self, data):
    """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    
    bag_of_words = []

    tokens = data.split()

    for i in tokens:
      bag_of_words.append((i, True))

    return bag_of_words


  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

  def __init__(self):
    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    pass

  def score(self, data):
    """
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
    pass

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    pass

  def featurize(self, data):
    """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    pass

  def __str__(self):
    return "NAME OF YOUR CLASSIFIER HERE"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  training_examples = generate_tuples_from_file(training)
  testing_examples = generate_tuples_from_file(testing)

  classifier = TextClassify()
  print(classifier)
  # do the things that you need to with your base class
  classifier.train(training_examples)
  gold_labels, predicted_labels = predict_examples(classifier, testing_examples)

  # report precision, recall, f1
  report_scores(gold_labels, predicted_labels)

  #-------

  #-------

  improved = TextClassifyImproved()
  print(improved)
  # do the things that you need to with your improved class


  # report final precision, recall, f1 (for your best model)




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
 








