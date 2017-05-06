from Document import Document
from sys import argv

if len(argv) != 2 and len(argv) != 3:
    print("Bad args")
    exit(-1)

if len(argv) == 3:
    end=int(argv[2])
else:
    end=-1

doc = Document('train_set.csv', end=end)
print(doc.data[int(argv[1])])
