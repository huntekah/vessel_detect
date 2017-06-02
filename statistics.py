def get_score(test_img, reference_img):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    all = 0
    for test_row, reference_row in zip(test_img, reference_img):
        for test_pixel, reference_pixel in zip(test_row, reference_row):
            if test_pixel == reference_pixel and test_pixel == 255:
                true_positive += 1
            if test_pixel == reference_pixel and test_pixel == 0:
                true_negative += 1
            if test_pixel != reference_pixel and test_pixel == 255:
                false_positive += 1
            if test_pixel != reference_pixel and test_pixel == 0:
                false_negative += 1
            all += 1
    return ((true_positive + true_negative) / all * 100,
            true_positive / (true_positive + false_negative) * 100,   # true positive rate (how good are ones)    czulosc
            true_negative / (false_positive + true_negative) * 100,   # true negative rate ( how good are zeros)  specyficznosc
            false_positive / (false_positive + true_negative)* 100)  # false positive rate
    # return (true_positive,
    # true_negative,
    # false_positive,
    # false_negative,
    # all)