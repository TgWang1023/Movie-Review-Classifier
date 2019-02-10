import sys

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