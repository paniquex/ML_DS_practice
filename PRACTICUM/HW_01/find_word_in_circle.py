def find_word_in_circle(circle, word):
    if len(circle) == 0:
        return -1
    word_size = len(word)
    circle_size = len(circle)
    circles_in_word = word_size // circle_size
    circle = circle * (circles_in_word + 2)
    circle_inv = circle[::-1]
    word_pos = circle.find(word)
    word_pos_inv = circle_inv.find(word)
    if word_pos != -1:
        return word_pos, 1
    elif word_pos_inv != -1:
        return circle_size - word_pos_inv - 1, -1
    return -1
