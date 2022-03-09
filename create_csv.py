import os
import yaml
import csv
from pathlib import Path

all_paths = []
for root, dirs, files in os.walk("evaluations/"):
    files = [file for file in files if ".yaml" in file]
    all_paths = [os.path.join(root, file) for file in files]

all_paths = sorted(all_paths)
print(all_paths)

rows = []
for path in all_paths:
    filename = path.split("/")[-1]
    model_name = filename.split(".")[0]
    with open(path) as fp:
        metrics_dict = yaml.load(fp, Loader=yaml.FullLoader)

    if not rows:
        metrics = [metric for metric in metrics_dict if metric != "avg_ref_self-BLEU2"]
        row = ["Model"]
        row += metrics
        rows.append(row)

    row = [model_name]
    for metric in metrics:
        if metric != "len_diff":
            row.append(round(metrics_dict[metric] * 100, 2))
        else:
            row.append(metrics_dict[metric])

    rows.append(row)


csv_filename = Path("evaluations/qgeval.csv")
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in rows:
        writer.writerow(row)


