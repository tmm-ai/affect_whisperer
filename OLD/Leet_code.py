# Is Unique: Implement an algorithm to determine if a string has all unique characters.
# What if you cannot use additional data structures?
from collections import Counter
def allun(string):
    # return len(string) == len(Counter(list(string)))
    # for pt1 in range(len(string)):
    #     for pt2 in range(pt1+1, len(string)):
    #         if string[pt1] == string[pt2]:
    #             return False
    # return True
    checker = [0]*128
    for pt1 in range(len(string)):
        letter = ord(string[pt1]) - ord("A")
        if checker[letter]: return False
        else:
            checker[letter] = 1
    return True






print(allun("theoperatTOXzmEbv"))
print(allun("theopraTOXzmEbv"))