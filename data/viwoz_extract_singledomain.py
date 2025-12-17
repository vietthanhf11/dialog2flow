import json
viwoz_dataset = json.load(open('viwoz/viwoz_2k8_data.json'))
len(viwoz_dataset)
viwoz_single = {}
for i, key in enumerate(viwoz_dataset):
  data = viwoz_dataset[key]
  data2 = {x:y for x, y in data['goal'].items() if len(y) > 0 and x not in ['message', 'topic']}
  if len(data2) == 1:
    data['goal'] = data2
    viwoz_single[key] = data
print(len(viwoz_single))
with open("viwoz/viwoz_1k5_single.json", "w") as v1k5:
  json.dump(viwoz_single, v1k5, indent=4, ensure_ascii=False)
