from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)

def compute_metrics(preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )