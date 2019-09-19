def parse_dict_line(dict_line):
    dict_line = dict_line.replace(' ', '')
    hyphen_idx = dict_line.find('-')
    human_word = dict_line[:hyphen_idx]
    dragon_words_list = dict_line[hyphen_idx+1:].split(',')
    return human_word, dragon_words_list


def generate_output(dragon_human_dict, output_dict_name):
    sorted_keys = sorted(dragon_human_dict)
    output_line = str(len(sorted_keys)) + '\n'
    for key in sorted_keys:
        output_line += str(key) + ' - '
        for value in sorted(dragon_human_dict[key]):
            output_line += value + ', '
        output_line = output_line[:-2]
        output_line += '\n'
    output_file = open(output_dict_name, 'w')
    output_file.write(output_line)
    output_file.close()


def get_new_dictionary(input_dict_name, output_dict_name):
    input_file = open(input_dict_name, 'r')
    lines_number = int(input_file.readline())
    dragon_human_dict = {}
    for i in range(lines_number):
        human_word, dragon_words_list = parse_dict_line(input_file.readline())
        for dragon_word in dragon_words_list:
            dragon_word = dragon_word.replace('\n', '')
            if dragon_word in dragon_human_dict.keys():
                dragon_human_dict[dragon_word].append(human_word)
            else:
                dragon_human_dict[dragon_word] = [human_word]
    input_file.close()
    generate_output(dragon_human_dict, output_dict_name)
