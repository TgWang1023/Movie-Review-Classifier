import sys
import copy
import math
import numpy as np
import scipy.stats as sp

# setting print options for testing purposes
# np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)

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
review = {}
pos_word_in_review_vector = {}
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
                if word not in review:
                    review[word] += 1
                total_pos_words += 1
    else:
        for w in review:
            if w in pos_word_in_review_vector:
                pos_word_in_review_vector[w] += 1
            else:
                pos_word_in_review_vector[w] = 1
total_pos_words += 1

# reading from negative training data
neg_dict = {}
review = {}
neg_word_in_review_vector = {}
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
                if word not in review:
                    review[word] += 1
                total_neg_words += 1
    else:
        for w in review:
            if w in neg_word_in_review_vector:
                neg_word_in_review_vector[w] += 1
            else:
                neg_word_in_review_vector[w] = 1
total_neg_words += 1

# combining data from both sets
total_dict = copy.deepcopy(pos_dict)
total_dict = dict.fromkeys(total_dict, 0)
total_words = total_pos_words + total_neg_words
for word, count in neg_dict.items():
    if word not in total_dict:
        total_dict[word] = 0


# (1) gaussian naive bayes classifier using bow feature
# calculating mean vector and variance vector for positive reviews
f1.seek(0)
# mean vector portion
pos_mean_vector = copy.deepcopy(pos_dict)
pos_mean_vector = dict.fromkeys(pos_mean_vector, 0)
pos_review_count = 0
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in pos_mean_vector:
                pos_mean_vector[word] += 1
    else:
        pos_review_count += 1
# add last review
pos_review_count += 1
# actual mean vector
for word in pos_mean_vector:
    pos_mean_vector[word] = pos_mean_vector[word] / pos_review_count
# variance vector portion
f1.seek(0)
pos_variance_vector = copy.deepcopy(pos_dict)
pos_variance_vector = dict.fromkeys(pos_variance_vector, 0)
pos_curr_vector = copy.deepcopy(pos_variance_vector)
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in pos_variance_vector:
                pos_curr_vector[word] += 1
    else:
        for w in pos_curr_vector:
            pos_variance_vector[w] += ((pos_curr_vector[w] - pos_mean_vector[w]) ** 2)
        pos_curr_vector = dict.fromkeys(pos_curr_vector, 0)
# add last review
for word in pos_curr_vector:
    pos_variance_vector[word] += ((pos_curr_vector[word] - pos_mean_vector[word]) ** 2)
# actual variance vector
for word in pos_variance_vector:
    pos_variance_vector[word] = pos_variance_vector[word] / pos_review_count

# calculating mean vector and variance vector for negative reviews
f2.seek(0)
# mean vector portion
neg_mean_vector = copy.deepcopy(neg_dict)
neg_mean_vector = dict.fromkeys(neg_mean_vector, 0)
neg_review_count = 0
for word in f2.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in neg_mean_vector:
                neg_mean_vector[word] += 1
    else:
        neg_review_count += 1
# add last review
neg_review_count += 1
# actual mean vector
for word in neg_mean_vector:
    neg_mean_vector[word] = neg_mean_vector[word] / neg_review_count
# variance vector portion
f2.seek(0)
neg_variance_vector = copy.deepcopy(neg_dict)
neg_variance_vector = dict.fromkeys(neg_variance_vector, 0)
neg_curr_vector = copy.deepcopy(neg_variance_vector)
for word in f2.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in neg_variance_vector:
                neg_curr_vector[word] += 1
    else:
        for w in neg_curr_vector:
            neg_variance_vector[w] += ((neg_curr_vector[w] - neg_mean_vector[w]) ** 2)
        neg_curr_vector = dict.fromkeys(neg_curr_vector, 0)
# add last review
for word in neg_curr_vector:
    neg_variance_vector[word] += ((neg_curr_vector[word] - neg_mean_vector[word]) ** 2)
# actual variance vector
for word in neg_variance_vector:
    neg_variance_vector[word] = neg_variance_vector[word] / neg_review_count

# recording accuracy for (1)
total_guesses_1 = 0
correct_guesses_1 = 0

# reading from positive test data
review = {}
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
                if w not in review:
                    review[w] = 1
                else:
                    review[w] += 1
    else:
        for word, occur in review.items():
            if word in pos_mean_vector:
                pos_mean = pos_mean_vector[word]
                pos_var = pos_variance_vector[word]  
                denom = (2 * math.pi * pos_var) ** 0.5
                num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                pos_result *= (num / denom)
            if word in neg_mean_vector:
                neg_mean = neg_mean_vector[word]
                neg_var = neg_variance_vector[word]
                denom = (2 * math.pi * neg_var) ** 0.5
                num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                neg_result *= (num / denom)
        if pos_result == 0.0 and neg_result == 0.0:
            for word, occur in review.items():
                if word in pos_mean_vector:
                    pos_mean = pos_mean_vector[word]
                    pos_var = pos_variance_vector[word]  
                    denom = (2 * math.pi * pos_var) ** 0.5
                    num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                    if num / denom != 0:
                        pos_result += math.log10(num / denom)
                if word in neg_mean_vector:
                    neg_mean = neg_mean_vector[word]
                    neg_var = neg_variance_vector[word]
                    denom = (2 * math.pi * neg_var) ** 0.5
                    num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                    if num / denom != 0:
                        neg_result += math.log10(num / denom)
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result)) 
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Positive")
            correct_guesses_1 += 1
        else:
            print("Predicted: Negative / Actual: Positve")
        total_guesses_1 += 1
        # reset parameters
        review = {}
        pos_result = 1.0
        neg_result = 1.0

# reading from negative test data
review = {}
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
                if w not in review:
                    review[w] = 1
                else:
                    review[w] += 1
    else:
        for word, occur in review.items():
            if word in pos_mean_vector:
                pos_mean = pos_mean_vector[word]
                pos_var = pos_variance_vector[word]  
                denom = (2 * math.pi * pos_var) ** 0.5
                num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                pos_result *= (num / denom)
            if word in neg_mean_vector:
                neg_mean = neg_mean_vector[word]
                neg_var = neg_variance_vector[word]
                denom = (2 * math.pi * neg_var) ** 0.5
                num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                neg_result *= (num / denom)
        if pos_result == 0.0 and neg_result == 0.0:
            for word, occur in review.items():
                if word in pos_mean_vector:
                    pos_mean = pos_mean_vector[word]
                    pos_var = pos_variance_vector[word]  
                    denom = (2 * math.pi * pos_var) ** 0.5
                    num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                    if num / denom != 0:
                        pos_result += math.log10(num / denom)
                if word in neg_mean_vector:
                    neg_mean = neg_mean_vector[word]
                    neg_var = neg_variance_vector[word]
                    denom = (2 * math.pi * neg_var) ** 0.5
                    num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                    if num / denom != 0:
                        neg_result += math.log10(num / denom)
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Negative")
        else:
            print("Predicted: Negative / Actual: Negative")
            correct_guesses_1 += 1
        total_guesses_1 += 1
        # reset parameters
        review = {}
        pos_result = 1.0
        neg_result = 1.0

###############################################################################################################
###############################################################################################################
###############################################################################################################

# (2) gaussian naive bayes classifier using tf-idf feature
# calculating mean vector and variance vector for positive reviews
f1.seek(0)
# mean vector portion
review = {}
pos_mean_vector = copy.deepcopy(pos_dict)
pos_mean_vector = dict.fromkeys(pos_mean_vector, 0.0)
pos_word_count = 0
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in pos_mean_vector:
                pos_word_count += 1
                if word not in review:
                    review[word] = 1
                else:
                    review[word] += 1
    else:
        for w in review:
            tf = review[w] / pos_word_count
            tmp1 = 0
            tmp2 = 0
            if w in pos_word_in_review_vector:
                tmp1 = pos_word_in_review_vector[w]
            if w in neg_word_in_review_vector:
                tmp2 = neg_word_in_review_vector[w]
            idf = math.log10((pos_review_count + neg_review_count) / (tmp1 + tmp2))
            if w in pos_mean_vector:
                pos_mean_vector[w] += (tf/idf)
        pos_word_count = 0
# actual mean vector
for word in pos_mean_vector:
    pos_mean_vector[word] = pos_mean_vector[word] / pos_review_count
# variance vector portion
f1.seek(0)
review = {}
pos_variance_vector = copy.deepcopy(pos_dict)
pos_variance_vector = dict.fromkeys(pos_variance_vector, 0)
pos_word_count = 0
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in pos_variance_vector:
                pos_word_count += 1
                if word not in review:
                    review[word] = 1
                else:
                    review[word] += 1
    else:
        for w in review:
            tf = review[w] / pos_word_count
            tmp1 = 0
            tmp2 = 0
            if w in pos_word_in_review_vector:
                tmp1 = pos_word_in_review_vector[w]
            if w in neg_word_in_review_vector:
                tmp2 = neg_word_in_review_vector[w]
            idf = math.log10((pos_review_count + neg_review_count) / (tmp1 + tmp2))
            if w in pos_variance_vector:
                pos_variance_vector[w] += (((tf/idf) - pos_mean_vector[w]) ** 2)
# actual variance vector
for word in pos_variance_vector:
    pos_variance_vector[word] = pos_variance_vector[word] / pos_review_count

# calculating mean vector and variance vector for negative reviews
f2.seek(0)
# mean vector portion
review = {}
neg_mean_vector = copy.deepcopy(pos_dict)
neg_mean_vector = dict.fromkeys(neg_mean_vector, 0.0)
neg_word_count = 0
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in neg_mean_vector:
                neg_word_count += 1
                if word not in review:
                    review[word] = 1
                else:
                    review[word] += 1
    else:
        for w in review:
            tf = review[w] / neg_word_count
            tmp1 = 0
            tmp2 = 0
            if w in pos_word_in_review_vector:
                tmp1 = pos_word_in_review_vector[w]
            if w in neg_word_in_review_vector:
                tmp2 = neg_word_in_review_vector[w]
            idf = math.log10((pos_review_count + neg_review_count) / (tmp1 + tmp2))
            if w in neg_mean_vector:
                neg_mean_vector[w] += (tf/idf)
        neg_word_count = 0
# actual mean vector
for word in neg_mean_vector:
    neg_mean_vector[word] = neg_mean_vector[word] / neg_review_count
# variance vector portion
f1.seek(0)
review = {}
neg_variance_vector = copy.deepcopy(pos_dict)
neg_variance_vector = dict.fromkeys(neg_variance_vector, 0)
pos_word_count = 0
for word in f1.read().split():
    if word != "/><br":
        if word not in stop_words:
            if word[-3:] == "<br":
                word = word[:-3]
            if word[:2] == "/>":
                word = word[2:]
            if word[-1:] in marks:
                word = word[:-1]
            if word[:1] in marks:
                word = word[1:]
            word = word.lower()
            if word in neg_variance_vector:
                neg_word_count += 1
                if word not in review:
                    review[word] = 1
                else:
                    review[word] += 1
    else:
        for w in review:
            tf = review[w] / neg_word_count
            tmp1 = 0
            tmp2 = 0
            if w in pos_word_in_review_vector:
                tmp1 = pos_word_in_review_vector[w]
            if w in neg_word_in_review_vector:
                tmp2 = neg_word_in_review_vector[w]
            idf = math.log10((pos_review_count + neg_review_count) / (tmp1 + tmp2))
            if w in neg_variance_vector:
                neg_variance_vector[w] += (((tf/idf) - neg_mean_vector[w]) ** 2)
# actual variance vector
for word in neg_variance_vector:
    neg_variance_vector[word] = neg_variance_vector[word] / neg_review_count

# recording accuracy for (2)
total_guesses_2 = 0
correct_guesses_2 = 0

# reading from positive test data
f3.seek(0)
review = {}
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
                if w not in review:
                    review[w] = 1
                else:
                    review[w] += 1
    else:
        for word, occur in review.items():
            if word in pos_mean_vector:
                pos_mean = pos_mean_vector[word]
                pos_var = pos_variance_vector[word]  
                denom = (2 * math.pi * pos_var) ** 0.5
                num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                pos_result *= (num / denom)
            if word in neg_mean_vector:
                neg_mean = neg_mean_vector[word]
                neg_var = neg_variance_vector[word]
                denom = (2 * math.pi * neg_var) ** 0.5
                num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                neg_result *= (num / denom)
        if pos_result == 0.0 and neg_result == 0.0:
            for word, occur in review.items():
                if word in pos_mean_vector:
                    pos_mean = pos_mean_vector[word]
                    pos_var = pos_variance_vector[word]  
                    denom = (2 * math.pi * pos_var) ** 0.5
                    num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                    if num / denom != 0:
                        pos_result += math.log10(num / denom)
                if word in neg_mean_vector:
                    neg_mean = neg_mean_vector[word]
                    neg_var = neg_variance_vector[word]
                    denom = (2 * math.pi * neg_var) ** 0.5
                    num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                    if num / denom != 0:
                        neg_result += math.log10(num / denom)
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result)) 
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Positive")
            correct_guesses_2 += 1
        else:
            print("Predicted: Negative / Actual: Positve")
        total_guesses_2 += 1
        # reset parameters
        review = {}
        pos_result = 1.0
        neg_result = 1.0

# reading from negative test data
f4.seek(0)
review = {}
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
                if w not in review:
                    review[w] = 1
                else:
                    review[w] += 1
    else:
        for word, occur in review.items():
            if word in pos_mean_vector:
                pos_mean = pos_mean_vector[word]
                pos_var = pos_variance_vector[word]  
                denom = (2 * math.pi * pos_var) ** 0.5
                num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                pos_result *= (num / denom)
            if word in neg_mean_vector:
                neg_mean = neg_mean_vector[word]
                neg_var = neg_variance_vector[word]
                denom = (2 * math.pi * neg_var) ** 0.5
                num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                neg_result *= (num / denom)
        if pos_result == 0.0 and neg_result == 0.0:
            for word, occur in review.items():
                if word in pos_mean_vector:
                    pos_mean = pos_mean_vector[word]
                    pos_var = pos_variance_vector[word]  
                    denom = (2 * math.pi * pos_var) ** 0.5
                    num = math.exp(-(float(occur)-float(pos_mean)) ** 2 / (2 * pos_var))
                    if num / denom != 0:
                        pos_result += math.log10(num / denom)
                if word in neg_mean_vector:
                    neg_mean = neg_mean_vector[word]
                    neg_var = neg_variance_vector[word]
                    denom = (2 * math.pi * neg_var) ** 0.5
                    num = math.exp(-(float(occur)-float(neg_mean)) ** 2 / (2 * neg_var))
                    if num / denom != 0:
                        neg_result += math.log10(num / denom)
        print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            print("Predicted: Positive / Actual: Negative")
        else:
            print("Predicted: Negative / Actual: Negative")
            correct_guesses_2 += 1
        total_guesses_2 += 1
        # reset parameters
        review = {}
        pos_result = 1.0
        neg_result = 1.0

###############################################################################################################
###############################################################################################################
###############################################################################################################

# (3) multinomial naive bayes classifier using bow feature
# recording accuracy for (3)
total_guesses_3 = 0
correct_guesses_3 = 0

# reading from positive test data
f3.seek(0)
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
        # print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            # print("Predicted: Positive / Actual: Positive")
            correct_guesses_3 += 1
        else:
            # print("Predicted: Negative / Actual: Positive")
            pass
        total_guesses_3 += 1
        # reset parameters
        review = []
        pos_result = 1.0
        neg_result = 1.0

# reading from negative test data
f4.seek(0)
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
        # print("Positive Probability: " + str(pos_result) + ", Negative Probability: " + str(neg_result))
        if pos_result >= neg_result:
            # print("Predicted: Positive / Actual: Negative")
            pass
        else:
            # print("Predicted: Negative / Actual: Negative")
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
# print("Gaussian Naive Bayes classifier using BoW feature:")
# print("-Total Prediction: " + str(total_guesses_1))
# print("-Correct Prediction: " + str(correct_guesses_1))
# print("-Accuracy: " + str(correct_guesses_1 / total_guesses_1))
print("---------------")
print("Gaussian Naive Bayes classifier using TF-IDF feature:")
print("-Total Prediction: " + str(total_guesses_2))
print("-Correct Prediction: " + str(correct_guesses_2))
print("-Accuracy: " + str(correct_guesses_2 / total_guesses_2))
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