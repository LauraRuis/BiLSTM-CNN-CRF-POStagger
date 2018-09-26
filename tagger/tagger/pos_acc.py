def get_pos_acc(gold, predicted, ud=False):

    correct = 0
    wrong = 0
    total = 0

    if ud:
        correct_upos = 0
        wrong_upos = 0
        total_upos = 0
    else:
        upos_acc = 0

    with open(gold, "r") as gold:
        with open(predicted, "r") as sys:

            for g, s in zip(gold, sys):
                g_line = g.split("\t")
                s_line = s.split("\t")
                if len(g_line) < 2 or len(s_line) < 2:
                    continue

                if not ud:
                    if g_line[0] != s_line[0]:
                        print("Gold and Predicted file not the same words")
                        return -1

                    g_pos = g_line[1]
                    s_pos = s_line[1]
                else:
                    if g_line[1] != s_line[1]:
                        print("Gold and Predicted file not the same words")
                        return -1
                    g_pos = g_line[4]
                    s_pos = s_line[4]
                    g_upos = g_line[3]
                    s_upos = s_line[3]

                total += 1
                if g_pos == s_pos:
                    correct += 1
                else:
                    wrong += 1

                if ud:
                    total_upos += 1
                    if g_upos == s_upos:
                        correct_upos += 1
                    else:
                        wrong_upos += 1

    if ud:
        upos_acc = (correct_upos / total_upos) * 100
    return (correct / total) * 100, upos_acc
