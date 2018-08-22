import matplotlib.pyplot as plt
import os
from collections import namedtuple
import re
import seaborn as sns

DIR = "logs_to_plot/"

all_lines = []
Line = namedtuple('line', ['name', 'iter', 'las', 'uas', 'xposacc', 'uposacc'])

for filename in os.listdir(DIR):
    if filename.startswith("plot"):
        path_to_file = os.path.join(DIR, filename)
        with open(path_to_file, "r") as infile:
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

            # print(filename)
            # print(filename.split(".")[0])
            save_name = filename.split(".")[0].split("plot_log_")[1]
            all_lines.append(Line(save_name, iter=x, las=las, uas=uas, xposacc=pos, uposacc=upos))


def plot_metric(fig, metric):

    ax = fig.add_subplot(111)

    if len(all_lines) <= 10:
        palette = sns.color_palette()
    else:
        palette = sns.color_palette("hls", 20)
    y_min = 100
    y_max = 0

    plt.title(metric)
    for i, l in enumerate(all_lines):

        color = palette[i]
        cur_line = getattr(l, metric)

        if cur_line:
            cur_min = min(cur_line)
            cur_max = max(cur_line)
            ax.plot(l.iter, cur_line, c=color, ls='-', label=l.name, fillstyle='none')
            if cur_min < y_min:
                y_min = cur_min
            if cur_max > y_max:
                y_max = cur_max
    ax.set_ylim(y_min-2, y_max+2)

    plt.xlabel("Iterations")
    plt.ylabel("%")
    plt.legend(loc=2)


metrics = ["uas", "las", "xposacc", "uposacc"]

for i, metric in enumerate(metrics):
    fig = plt.figure(i)
    plot_metric(fig, metric)
plt.show()



