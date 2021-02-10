import os
import shutil
import random

seed = 1
random.seed(seed)
directory = "C:/Users/youss/Desktop/isic/images/"
train = "C:/Users/youss/Desktop/isic/data/train/"
test = "C:/Users/youss/Desktop/isic/data/test/"
validation = "C:/Users/youss/Desktop/isic/data/validation/"


os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")
os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")
os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

test_examples = train_examples = validation_examples = 0
#read csv file
for line in open("C:/Users/youss/Desktop/isic/labels.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malign = split_line[1]

    random_num = random.random()

    if random_num < 0.8:
        location = train
        train_examples += 1

    elif random_num < 0.9:
        location = validation
        validation_examples += 1

    else:
        location = test
        test_examples +=1

    if int(float(benign_malign)) == 0:
        shutil.copy("C:/Users/youss/Desktop/isic/images/" + img_file + ".jpg",
                    location + "benign/" + img_file + ".jpg",

        )
    elif int(float(benign_malign)) == 1:
        shutil.copy("C:/Users/youss/Desktop/isic/images/" + img_file + ".jpg",
                    location + "malignant/" + img_file + ".jpg",

        )

print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")
print(f"Number of validation examples {validation_examples}")



