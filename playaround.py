import pickle

file = open('Assignment/Q1_data/data.pkl', 'rb')
data =  pickle.load(file)
file.close()

count = 0 
for item in data:
    print(item)
    count = count + 1

print("There are ", count, "entries!")