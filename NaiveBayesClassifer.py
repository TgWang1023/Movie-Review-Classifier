import sys
import copy
import math

# check inputs and initlialize files
if len(sys.argv) != 5:
    print("Invalid number of arguments. Follow the format: NaiveBayesClassifer.py *POS TRAINING FILE* *NEG TRAINING FILE* *POS TEST FILE* *NEG TEST FILE*")
    exit(0)
f1 = open(sys.argv[1], 'r')
f2 = open(sys.argv[2], 'r')
f3 = open(sys.argv[3], 'r')
f4 = open(sys.argv[4], 'r')

# multinomial naive bayes classifier

# common puncuation marks
marks = {",", ":", "'", ".", "-", "!", "?", ";"}

# reading from positive training data
pos_dict = {}
total_pos_words = 0
for word in f1.read().split():
    # print(word)
    if word != "/><br":
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
        if word[-1:] in marks:
            word = word[:-1]
        if word in pos_dict:
            pos_dict[word] += 1
        else:
            pos_dict[word] = 1
        total_pos_words += 1
total_pos_words += 1
    
# reading from negative training data
neg_dict = {}
total_neg_words = 0
for word in f2.read().split():
    # print(word)
    if word != "/><br":
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
        if word[-1:] in marks:
            word = word[:-1]
        if word in neg_dict:
            neg_dict[word] += 1
        else:
            neg_dict[word] = 1
        total_neg_words += 1
total_neg_words += 1

# # combining data from both sets
# total_dict = copy.deepcopy(pos_dict)
# total_words = total_pos_words + total_neg_words
# for word, count in neg_dict.items():
#     if word in total_dict:
#         total_dict[word] += count
#     else:
#         total_dict[word] = count

# recording accuracy
total_guesses = 0
correct_guesses = 0

# reading from positive test data
review = []
pos_result = 1.0
neg_result = 1.0
for w in f3.read().split():
    # print(w)
    if w != "/><br":
        if w[-3:] == "<br":
            w = w[:-3]
        if w[:2] == "/>":
            w = w[2:]
        review.append(w)
    else:
        for word in review:
            if word in pos_dict:
                new_pos = pos_result * (pos_dict[word] / total_pos_words)
                # while new_pos <= 0:
                #     pos_result = math.log10(pos_result)
                #     new_pos = pos_result * (pos_dict[word] / total_pos_words)
                pos_result = new_pos
            else:
                # laplace smoothing
                new_pos = pos_result * (1 / total_pos_words)
                # while new_pos <= 0:
                #     pos_result = math.log10(pos_result)
                #     new_pos = pos_result * (pos_dict[word] / total_pos_words)
                pos_result = new_pos
            if word in neg_dict:
                new_neg = neg_result * (neg_dict[word] / total_neg_words)
                # while new_neg <= 0:
                #     neg_result = math.log10(neg_result)
                #     new_neg = neg_result * (neg_dict[word] / total_neg_words)
                neg_result = new_neg
            else:
                # laplace smoothing
                new_neg = neg_result * (1 / total_neg_words)
                # while new_neg <= 0:
                #     neg_result = math.log10(neg_result)
                #     new_neg = neg_result * (neg_dict[word] / total_neg_words)
                neg_result = new_neg
            print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
            if pos_result >= neg_result:
                print("Predicted: Positive / Actual: Positive")
                correct_guesses += 1
            else:
                print("Predicted: Negative / Actual: Positive")
            total_guesses += 1
            # reset parameters
            review = []
            pos_result = 1.0
            neg_result = 1.0

# reading from negative test data
review = []
pos_result = 1.0
neg_result = 1.0
for w in f4.read().split():
    # print(w)
    if w != "/><br":
        if w[-3:] == "<br":
            w = w[:-3]
        if w[:2] == "/>":
            w = w[2:]
        review.append(w)
    else:
        for word in review:
            if word in pos_dict:
                new_pos = pos_result * (pos_dict[word] / total_pos_words)
                # while new_pos <= 0:
                #     pos_result = math.log10(pos_result)
                #     new_pos = pos_result * (pos_dict[word] / total_pos_words)
                pos_result = new_pos
            else:
                # laplace smoothing
                new_pos = pos_result * (1 / total_pos_words)
                # while new_pos <= 0:
                #     pos_result = math.log10(pos_result)
                #     new_pos = pos_result * (pos_dict[word] / total_pos_words)
                pos_result = new_pos
            if word in neg_dict:
                new_neg = neg_result * (neg_dict[word] / total_neg_words)
                # while new_neg <= 0:
                #     neg_result = math.log10(neg_result)
                #     new_neg = neg_result * (neg_dict[word] / total_neg_words)
                neg_result = new_neg
            else:
                # laplace smoothing
                new_neg = neg_result * (1 / total_neg_words)
                # while new_neg <= 0:
                #     neg_result = math.log10(neg_result)
                #     new_neg = neg_result * (neg_dict[word] / total_neg_words)
                neg_result = new_neg
            print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
            if pos_result >= neg_result:
                print("Predicted: Positive / Actual: Negative")
            else:
                print("Predicted: Negative / Actual: Negative")
                correct_guesses += 1
            total_guesses += 1
            # reset parameters
            review = []
            pos_result = 1.0
            neg_result = 1.0

# print final result
print("---------------")
print("Final Result: ")
print("Total Prediction: " + str(total_guesses))
print("Correct Prediction: " + str(correct_guesses))
print("Accuracy: " + str(correct_guesses / total_guesses))