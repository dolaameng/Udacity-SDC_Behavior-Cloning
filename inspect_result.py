import argparse
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=str, dest="result_file", help="generated result file", 
					default="tmp/test_result.csv")
args = parser.parse_args()
print(args.result_file)


df = pd.read_csv(args.result_file)
df["error"] = np.abs(df["steer"]-df["prediction"])
df = df.sort_values(by="error")

def show_top_mistakes(n=5):
	for r in df.tail(n).to_records():
		im = Image.open(r.image)
		plt.imshow(im)
		title = "real: %.2f vs pred: %.2f" % (r.steer, r.prediction)
		plt.title(title)
		plt.show()

def show_top_best(n=5):
	for r in df.head(n).to_records():
		im = Image.open(r.image)
		plt.imshow(im)
		title = "real: %.2f vs pred: %.2f" % (r.steer, r.prediction)
		plt.title(title)
		plt.show()

def scatter_plot():
	ax = df.plot(kind="scatter", x="steer", y="prediction", marker="o")
	ax.plot(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01), c="r")
	plt.show()

if __name__ == "__main__":
	scatter_plot()