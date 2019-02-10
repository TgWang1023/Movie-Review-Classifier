import sys
import copy

# Multinomial Naive Bayes Classifier
# reading from positive training data
pos_dict = {}
total_pos_words = 0
f1 = open(sys.argv[1], 'r')
for word in f1.read().split():
    # print(word)
    if word != "/><br":
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
        if word in pos_dict:
            pos_dict[word] += 1
        else:
            pos_dict[word] = 1
        total_pos_words += 1
total_pos_words += 1
    
# reading from negative training data
neg_dict = {}
total_neg_words = 0
f2 = open(sys.argv[2], 'r')
for word in f2.read().split():
    # print(word)
    if word != "/><br":
        if word[-3:] == "<br":
            word = word[:-3]
        if word[:2] == "/>":
            word = word[2:]
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

print(len(pos_dict))
print(len(neg_dict))
print(len(total_dict))