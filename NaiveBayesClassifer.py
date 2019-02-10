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

# common puncuation marks
marks = {",", ":", "'", ".", "-", "!", "?", ";", "(", ")"}
# stop words
stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", 
    "having", "he", "he’d", "he’ll", "he’s", "her", "here", "here’s", "hers", "herself", "him", "himself", "his", "how", "how’s", "I", "I’d", "I’ll", "I’m", "I’ve", 
    "if", "in", "into", "is", "it", "it’s", "its", "itself", "let’s", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", 
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she’d", "she’ll", "she’s", "should", "so", "some", "such", "than", "that", "that’s", "the", "their", 
    "theirs", "them", "themselves", "then", "there", "there’s", "these", "they", "they’d", "they’ll", "they’re", "they’ve", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "we’d", "we’ll", "we’re", "we’ve", "were", "what", "what’s", "when", "when’s", "where", "where’s", "which", "while", "who", "who’s", 
    "whom", "why", "why’s", "with", "would", "you", "you’d", "you’ll", "you’re", "you’ve", "your", "yours", "yourself", "yourselves"}

# (1) gaussian naive bayes classifier using bow feature


# (2) gaussian navie bayes classifier using tf-idf feature


# (3) multinomial naive bayes classifier using bow feature
# reading from positive training data
pos_dict = {}
total_pos_words = 0
for word in f1.read().split():
    # print(word)
    if word != "/><br" and word not in stop_words:
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
        if word[-1:] in marks:
            word = word[:-1]
        if word[:1] in marks:
            word = word[1:]
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
    if word != "/><br" and word not in stop_words:
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
        if word[-1:] in marks:
            word = word[:-1]
        if word[:1] in marks:
            word = word[1:]
        if word in neg_dict:
            neg_dict[word] += 1
        else:
            neg_dict[word] = 1
        total_neg_words += 1
total_neg_words += 1

# combining data from both sets
total_dict = copy.deepcopy(pos_dict)
total_words = total_pos_words + total_neg_words
for word, count in neg_dict.items():
    if word in total_dict:
        total_dict[word] += count
    else:
        total_dict[word] = count

# recording accuracy for (3)
total_guesses_3 = 0
correct_guesses_3 = 0

# reading from positive test data
review = []
pos_result = 1.0
neg_result = 1.0
for w in f3.read().split():
    # print(w)
    if w != "/><br":
        if w not in stop_words:
            if w[-3:] == "<br":
                w = w[:-3]
            if w[:2] == "/>":
                w = w[2:]
            if w[-1:] in marks:
                w = w[:-1]
            if w[:1] in marks:
                w = w[1:]
            review.append(w)
    else:
        for word in review:
            if word in pos_dict:
                pos_result = pos_result * ((pos_dict[word] + 1) / (total_pos_words + len(total_dict)))
            else:
                # laplace smoothing
                pos_result = pos_result * (1 / (total_pos_words + len(total_dict)))
            if word in neg_dict:
                neg_result = neg_result * ((neg_dict[word] + 1) / (total_neg_words + len(total_dict)))
            else:
                # laplace smoothing
                neg_result = neg_result * (1 / (total_neg_words + len(total_dict)))
        # result underflows
        if pos_result == 0 and neg_result == 0:
            for word in review:
                if word in pos_dict:
                    pos_result = pos_result + math.log10((pos_dict[word] + 1) / (total_pos_words + len(total_dict)))
                else:
                    # laplace smoothing
                    pos_result = pos_result + math.log10(1 / (total_pos_words + len(total_dict)))
                if word in neg_dict:
                    neg_result = neg_result + math.log10((neg_dict[word] + 1) / (total_neg_words + len(total_dict)))
                else:
                    # laplace smoothing
                    neg_result = neg_result + math.log10(1 / (total_neg_words + len(total_dict)))
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Positive")
            correct_guesses_3 += 1
        else:
            print("Predicted: Negative / Actual: Positive")
        total_guesses_3 += 1
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
        if w not in stop_words:
            if w[-3:] == "<br":
                w = w[:-3]
            if w[:2] == "/>":
                w = w[2:]
            if w[-1:] in marks:
                w = w[:-1]
            if w[:1] in marks:
                w = w[1:]
            review.append(w)
    else:
        for word in review:
            if word in pos_dict:
                pos_result = pos_result * ((pos_dict[word] + 1) / (total_pos_words + len(total_dict)))
            else:
                # laplace smoothing
                pos_result = pos_result * (1 / (total_pos_words + len(total_dict)))
            if word in neg_dict:
                neg_result = neg_result * ((neg_dict[word] + 1) / (total_neg_words + len(total_dict)))
            else:
                # laplace smoothing
                neg_result = neg_result * (1 / (total_neg_words + len(total_dict)))
        # result underflows
        if pos_result == 0.0 and neg_result == 0.0:
            for word in review:
                if word in pos_dict:
                    pos_result = pos_result + math.log10((pos_dict[word] + 1) / (total_pos_words + len(total_dict)))
                else:
                    # laplace smoothing
                    pos_result = pos_result + math.log10(1 / (total_pos_words + len(total_dict)))
                if word in neg_dict:
                    neg_result = neg_result + math.log10((neg_dict[word] + 1) / (total_neg_words + len(total_dict)))
                else:
                    # laplace smoothing
                    neg_result = neg_result + math.log10(1 / (total_neg_words + len(total_dict)))
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Negative")
        else:
            print("Predicted: Negative / Actual: Negative")
            correct_guesses_3 += 1
        total_guesses_3 += 1
        # reset parameters
        review = []
        pos_result = 1.0
        neg_result = 1.0

# print final result
print("---------------")
print("Final Result: ")
print("---------------")
print("Gaussian Naive Bayes classifier using BoW feature:")
print("-Total Prediction: " + str(total_guesses_3))
print("-Correct Prediction: " + str(correct_guesses_3))
print("-Accuracy: " + str(correct_guesses_3 / total_guesses_3))
print("---------------")
print("Gaussian Naive Bayes classifier using TF-IDF feature:")
print("-Total Prediction: " + str(total_guesses_3))
print("-Correct Prediction: " + str(correct_guesses_3))
print("-Accuracy: " + str(correct_guesses_3 / total_guesses_3))
print("---------------")
print("Multinomial Naive Bayes classifier using BoW feature:")
print("-Total Prediction: " + str(total_guesses_3))
print("-Correct Prediction: " + str(correct_guesses_3))
print("-Accuracy: " + str(correct_guesses_3 / total_guesses_3))
print("\n")