import math

def calculate_data(name, vp, vn, fp, fn):

    print(f"Name: {name}")
    print(f"True Positive: {vp}")
    print(f"True Negative: {vn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")

    up = (vp * vn) - (fp * fn)
    down = (vp + fn) * (vp + fp) * (vn + fp) * (vn + fn)

    ccm = up / math.sqrt(down)
    print(f"CCM: {ccm}")

    acc = (vp + vn) / (vp + vn + fp + fn)
    print(f"ACC: {acc}")

    prec = vp / (vp+fp)
    print(f"PRE: {prec}")

    recall = vp / (vp + fn)
    print(f"Recall: {recall}")

    f1 = 2* ((prec*recall) / (prec + recall))
    print(f"F1: {f1}")


if __name__ == '__main__':
    calculate_data("name", 881, 2827, 322, 663)