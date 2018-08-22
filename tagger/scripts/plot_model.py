import matplotlib.pyplot as plt
import os
from collections import namedtuple
import re
import seaborn as sns
import sys

f1 = sys.argv[1]
if len(sys.argv) > 2:
    f2 = sys.argv[2]
else:
    f2 = False


def get_line(file):
    Line = namedtuple('line', ['name', 'iter', 'las', 'uas', 'xposacc', 'uposacc'])

    with open(file, "r") as infile:
        las, uas, pos, upos = [], [], [], []
        x = []
        for line in infile:
            data = line.split(" ")
            if len(data) > 1:
                # print(data)
                if data[0] == "Iter":
                    x.append(int(''.join(filter(str.isdigit, line))))
                if data[0] == "LAS":
                    las.append(float(re.findall("\d+\.\d+", line)[0])*100)
                if data[0] == "UAS":
                    uas.append(float(re.findall("\d+\.\d+", line)[0])*100)
                if data[0] == "XPOS-ACC":
                    pos.append(float(re.findall("\d+\.\d+", line)[0])*100)
                if data[0] == "UPOS-ACC":
                    upos.append(float(re.findall("\d+\.\d+", line)[0])*100)
        lines = Line(file.split(".")[0].split("plot_log_")[1], iter=x, las=las, uas=uas, xposacc=pos, uposacc=upos)

    return lines


lines1 = get_line(f1)
if f2:
    lines2 = get_line(f2)
    lines = [lines1, lines2]
else:
    lines = [lines1]


def plot_model(lines):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    palette = sns.color_palette()

    y_min = 100
    y_max = 0

    metrics = ["uas", "las", "xposacc", "uposacc"]
    title = ""
    styles = ["-", "--"]

    for j, line in enumerate(lines):
        title = title + " and " + line.name + " as " + styles[j]
        for i, metric in enumerate(metrics):
            color = palette[i]
            cur_line = getattr(line, metric)

            if cur_line:
                cur_min = min(cur_line)
                cur_max = max(cur_line)
                ax.plot(line.iter, cur_line, c=color, ls=styles[j], label=metric, fillstyle='none')
                if cur_min < y_min:
                    y_min = cur_min
                if cur_max > y_max:
                    y_max = cur_max
    ax.set_ylim(y_min-2, y_max+2)

    plt.title(title[5:])
    plt.xlabel("Iterations")
    plt.ylabel("%")
    plt.legend(loc=2)
    plt.show()


plot_model(lines)