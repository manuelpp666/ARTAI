import glob
import os

keywords = ["Arte", "Pintura", "Escultura", "Museo", "Historia del arte"]

output_file = "dataset/wiki_arte.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as out_f:
    for file in glob.glob("extracted/*/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            article = ""
            for line in f:
                if line.strip() == "":
                    if any(k.lower() in article.lower() for k in keywords):
                        out_f.write(article + "\n\n")
                    article = ""
                else:
                    article += line
