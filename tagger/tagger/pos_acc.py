def get_pos_acc(gold, predicted):

    correct = 0
    wrong = 0
    total = 0
    with open(gold, "r") as gold:
        with open(predicted, "r") as sys:
            for g, s in zip(gold, sys):
                g_line = g.split("\t")
                s_line = s.split("\t")
                if len(g_line) < 2 or len(s_line) < 2:
                    continue

                if g_line[0] != s_line[0]:
                    print("Gold and Predicted file not the same words")
                    return -1

                g_pos = g_line[1]
                s_pos = s_line[1]
                total += 1
                if g_pos == s_pos:
                    correct += 1
                else:
                    wrong += 1
    return (correct / total) * 100
