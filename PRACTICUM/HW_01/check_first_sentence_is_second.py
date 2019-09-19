def check_first_sentence_is_second(s1, s2):
    s1_list = s1.split()
    s2_list = s2.split()
    result = False
    if (len(s1_list) < len(s2_list)):
        result = False
    else:
        sum_occurrences = sum([word in s1_list for word in s2_list])
        result = sum_occurrences == len(s2_list)
    return result
