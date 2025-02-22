import csv
import time

epochs = 10
with open('metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])

    for e in range(epochs):
        writer.writerow([str(e), "0.1", "0.2", "0.3", "0.4"])
        print(f"train epoch {e}")
        time.sleep(1)

