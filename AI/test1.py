import json

coco = json.load(open("../dataset/merged_training.json"))

print("Categories:")
for c in coco["categories"]:
    print(c)

print("\nSample annotation:")
print(coco["annotations"][0])
