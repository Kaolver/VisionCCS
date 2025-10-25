import json

dataset = json.load(open('vqav2_filtered_categorized_checkpoint_7200.json', 'r'))

print(dataset['object_detection'][0])

# image_id is question_id // 1000
for category in dataset:
    for item in dataset[category]:
        image_id = item['question_id'] // 1000
        leading_zeros = 12 - len(str(image_id))
        item['image_id'] = '0' * leading_zeros + str(image_id) + '.jpg'


print(dataset['object_detection'][0])

json.dump(dataset, open('vqav2_filtered_categorized_checkpoint_7200_fixed.json', 'w'), indent=2)
