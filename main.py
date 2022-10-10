import numpy as np
import pandas as pd
import plotly.express as px

fb_players = np.random.randn(500) * 20 + 160  # negative class - 0
bb_players = np.random.randn(500) * 10 + 190  # positive class - 1


def height_classificator(height, players):
    return [0 if pl < height else 1 for pl in players]


def accuracy(neg, pos):
    N = len(neg) + len(pos)
    TP = pos.count(1)
    TN = neg.count(0)
    return (TP + TN) / N


def precision(neg, pos):
    TP = pos.count(1)
    if (TP == 0):
        return 1
    FP = neg.count(1)
    return TP / (TP + FP)


def recall(pos):
    TP = pos.count(1)
    FN = pos.count(0)
    return TP / (TP + FN)


def one_spe(neg):
    FP = neg.count(1)
    TN = neg.count(0)
    return FP / (FP + TN)


def sen(pos):
    TP = pos.count(1)
    FN = pos.count(0)
    return TP / (TP + FN)


def create_df(data, columns):
    return pd.DataFrame(data=data, columns=columns)


def create_PR(x, y, th, ac, title, name=None, path2save=None):
    df = create_df(np.stack((x, y, th, ac), axis=1),
                   ['recall', 'precision', 'threshold', 'accuracy'])
    # pd.set_option('display.max_rows', None)
    # print(df)
    fig = px.line(df, x="recall", y="precision", hover_data=['threshold', 'accuracy'])
    fig.update_layout(title=title)
    fig.update_traces(mode="markers+lines")
    fig.show()
    fig.write_html(f"{path2save}/{name}.html")
    return

def trap_AUC(x, y):
    h = [x[i+1]-x[i] for i in range(len(x)-1)]
    trap_s = [(y[i]+y[i+1])*h[i]/2 for i in range(len(h))]
    return sum(trap_s)

def create_ROC(x, y, title, name=None, path2save=None):
    df = create_df(np.stack((x, y), axis=1), ['1-specificity', 'sensitivity'])
    fig = px.line(df, x="1-specificity", y="sensitivity")
    fig.update_layout(title=f"{title}, AUC = {trap_AUC(x[::-1], y[::-1])}")
    fig.update_traces(mode="markers+lines")
    fig.show()
    fig.write_html(f"{path2save}/{name}.html")


thresholds = [i for i in range(0, 231)]

recalls = []  # x
precisions = []  # y
accuracies = []

one_spes = []  # x
sens = []  # y

for i in range(0, 231):
    fb_pl_cl = height_classificator(i, fb_players)  # neg
    bb_pl_cl = height_classificator(i, bb_players)  # pos
    # Precision-Recall
    recalls.append(recall(bb_pl_cl))
    precisions.append(precision(fb_pl_cl, bb_pl_cl))
    accuracies.append(accuracy(fb_pl_cl, bb_pl_cl))
    # ROC-Curve
    one_spes.append(one_spe(fb_pl_cl))
    sens.append(sen(bb_pl_cl))
create_PR(recalls, precisions, thresholds, accuracies, 'Precision-Recall Curve', name="Presicion_Recall",
          path2save="C:/Users/26067/PycharmProjects/ML_HW4_classification")
create_ROC(one_spes, sens, 'ROC-Curve', name="ROC_Curve",
           path2save="C:/Users/26067/PycharmProjects/ML_HW4_classification")


