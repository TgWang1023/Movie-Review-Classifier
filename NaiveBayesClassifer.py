import sys
import copy
import math
import numpy as np

# setting print options for testing purposes
# np.set_printoptions(threshold = np.inf)

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
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", 
    "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", 
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", 
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", 
    "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"}

print("Training data, please wait patiently...")
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
        if word[:1] in marks:
            word = word[1:]
        if word.isalpha():
            word = word.lower()
            if word not in stop_words:
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
        if word[:1] in marks:
            word = word[1:]
        if word.isalpha():
            word = word.lower()
            if word not in stop_words:
                if word in neg_dict:
                    neg_dict[word] += 1
                else:
                    neg_dict[word] = 1
                total_neg_words += 1
total_neg_words += 1

# combining data from both sets
total_dict = copy.deepcopy(pos_dict)
total_dict = dict.fromkeys(total_dict, 0)
total_words = total_pos_words + total_neg_words
for word, count in neg_dict.items():
    if word not in total_dict:
        total_dict[word] = 0


# # (1) gaussian naive bayes classifier using bow feature
# # calculating mean vector and covariance matrix for positive reviews
# f1.seek(0)
# # mean vector portion
# pos_mean_vector = np.array([0] * len(total_dict))
# pos_current_dict = copy.deepcopy(total_dict)
# pos_review_count = 0
# for word in f1.read().split():
#     if word != "/><br":
#         if word not in stop_words:
#             if word[-3:] == "<br":
#                 word = word[:-3]
#             if word[:2] == "/>":
#                 word = word[2:]
#             if word[-1:] in marks:
#                 word = word[:-1]
#             if word[:1] in marks:
#                 word = word[1:]
#             if word in pos_current_dict:
#                 pos_current_dict[word] += 1
#     else:
#         pos_current_vector = np.array(list(pos_current_dict.values()))
#         pos_mean_vector = pos_mean_vector + pos_current_vector
#         pos_review_count += 1
#         pos_current_dict = dict.fromkeys(pos_current_dict, 0)
# # add last review
# pos_current_vector = np.array(list(pos_current_dict.values()))
# pos_mean_vector = pos_current_vector + pos_mean_vector
# pos_review_count += 1
# # actual mean vector
# pos_mean_vector = pos_mean_vector / pos_review_count
# print(pos_mean_vector)
# # covariance matrix portion
# f1.seek(0)
# pos_current_dict = copy.deepcopy(total_dict)
# pos_cov_mtx = np.zeros(shape = (len(total_dict), len(total_dict)))
# for word in f1.read().split():
#     if word != "/><br":
#         if word not in stop_words:
#             if word[-3:] == "<br":
#                 word = word[:-3]
#             if word[:2] == "/>":
#                 word = word[2:]
#             if word[-1:] in marks:
#                 word = word[:-1]
#             if word[:1] in marks:
#                 word = word[1:]
#             if word in pos_current_dict:
#                 pos_current_dict[word] += 1
#     else:
#         pos_current_vector = np.array(list(pos_current_dict.values()))
#         pos_diff_vector = pos_current_vector - pos_mean_vector
#         pos_trans_vector = np.transpose(pos_diff_vector)
#         pos_cov_mtx += pos_diff_vector * pos_trans_vector
#         pos_current_dict = dict.fromkeys(pos_current_dict, 0)
# # add last review
# pos_current_vector = np.array(list(pos_current_dict.values()))
# pos_diff_vector = pos_current_vector - pos_mean_vector
# pos_trans_vector = np.transpose(pos_diff_vector)
# pos_cov_mtx = pos_cov_mtx / pos_review_count
# print(pos_cov_mtx)


# # calculating mean vector and covariance matrix for negative reviews
# f2.seek(0)
# # mean vector portion
# neg_mean_vector = np.array([0] * len(total_dict))
# neg_current_dict = copy.deepcopy(total_dict)
# neg_review_count = 0
# for word in f2.read().split():
#     if word != "/><br":
#         if word not in stop_words:
#             if word[-3:] == "<br":
#                 word = word[:-3]
#             if word[:2] == "/>":
#                 word = word[2:]
#             if word[-1:] in marks:
#                 word = word[:-1]
#             if word[:1] in marks:
#                 word = word[1:]
#             if word in neg_current_dict:
#                 neg_current_dict[word] += 1
#     else:
#         neg_current_vector = np.array(list(neg_current_dict.values()))
#         neg_mean_vector = np.add(neg_current_vector, neg_mean_vector)
#         neg_review_count += 1
#         neg_current_dict = dict.fromkeys(neg_current_dict, 0)
# # add last review
# neg_current_vector = np.array(list(neg_current_dict.values()))
# neg_mean_vector = np.add(neg_current_vector, neg_mean_vector)
# neg_review_count += 1
# # actual mean vector
# neg_mean_vector = neg_mean_vector / neg_review_count
# # covariance matrix portion
# f2.seek(0)
# neg_current_dict = copy.deepcopy(total_dict)
# neg_cov_mtx = np.zeros(shape = (len(total_dict), len(total_dict)))
# for word in f1.read().split():
#     if word != "/><br":
#         if word not in stop_words:
#             if word[-3:] == "<br":
#                 word = word[:-3]
#             if word[:2] == "/>":
#                 word = word[2:]
#             if word[-1:] in marks:
#                 word = word[:-1]
#             if word[:1] in marks:
#                 word = word[1:]
#             if word in neg_current_dict:
#                 neg_current_dict[word] += 1
#     else:
#         neg_current_vector = np.array(list(neg_current_dict.values()))
#         neg_diff_vector = neg_current_vector - neg_mean_vector
#         neg_trans_vector = np.transpose(neg_diff_vector)
#         neg_cov_mtx += neg_diff_vector * neg_trans_vector
#         neg_current_dict = dict.fromkeys(neg_current_dict, 0)
# # add last review
# neg_current_vector = np.array(list(neg_current_dict.values()))
# neg_diff_vector = neg_current_vector - neg_mean_vector
# neg_trans_vector = np.transpose(neg_diff_vector)
# neg_cov_mtx = neg_cov_mtx / neg_review_count
# print(neg_cov_mtx)


# recording accuracy for (1)
total_guesses_1 = 0
correct_guesses_1 = 0



# # (2) gaussian naive bayes classifier using tf-idf feature
# # recording accuracy for (2)
# total_guesses_2 = 0
# correct_guesses_2 = 0

# (3) multinomial naive bayes classifier using bow feature
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
        if w[-3:] == "<br":
            w = w[:-3]
        if w[:2] == "/>":
            w = w[2:]
        if w[-1:] in marks:
            w = w[:-1]
        if w[:1] in marks:
            w = w[1:]
        if w.isalpha():
            w = w.lower()
            if w not in stop_words:
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
        if w[-3:] == "<br":
            w = w[:-3]
        if w[:2] == "/>":
            w = w[2:]
        if w[-1:] in marks:
            w = w[:-1]
        if w[:1] in marks:
            w = w[1:]
        if w.isalpha():
            w = w.lower()
            if w not in stop_words:
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

# close files
f1.close()
f2.close()
f3.close()
f4.close()