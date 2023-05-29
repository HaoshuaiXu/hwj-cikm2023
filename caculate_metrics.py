from sklearn.metrics import precision_score, recall_score, f1_score


def caculate_output_metrics(y_true:list, y_pred:list, epoch:int):
    pre = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    with open("./result_log/result_log.tsv", "+a") as f:
        f.writelines(f"{epoch}\t{pre}\t{rec}\t{f1}\n")