import math


def get_all_dividors(number):
    dividor_list = []
    i = 1
    while i <= math.sqrt(number):
        if number % i == 0:
            dividor_list.append(i)
            dividor_list.append(int(number / i))
        i += 1
    return dividor_list


def find_max_substring_occurrence(input_string):
    substr_possible_len_list = sorted(get_all_dividors(len(input_string)))
    for substr_len in substr_possible_len_list:
        substr = input_string[:substr_len]
        substr_occurrence_number = len(input_string) // substr_len
        substr_multiplied = substr * substr_occurrence_number
        if substr_multiplied == input_string:
            return substr_occurrence_number
    return 0
