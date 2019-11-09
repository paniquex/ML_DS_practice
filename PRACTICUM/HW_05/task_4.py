def WordContextGenerator(words, window_size):
    for i, word in enumerate(words):
        for left_word in words[max(i - window_size, 0):i]:
            yield word, left_word
        for right_word in words[i+1: i + window_size + 1]:
            yield word, right_word
