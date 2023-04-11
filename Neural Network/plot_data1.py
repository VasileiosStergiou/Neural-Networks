import matplotlib.pyplot as plt
from os import system

system("javac *.java")
system("java Main1")
filename="outputs.txt"
f = open(filename)

filename1="test_set.txt"
f1 = open(filename1)

colors=["r","g","b","y"]

correct_x=[]
correct_y=[]
correct_c=[]

incorrect_x=[]
incorrect_y=[]
incorrect_c=[]

correct_categories = []

for line1,line2 in zip(f.readlines(),f1.readlines()):
	x=[float(x) for x in line1.strip().split(",")]
	category =[float(x) for x in line2.strip().split(",")][2]

	if (x[2] == category):
		correct_x.append(x[0])
		correct_y.append(x[1])
		correct_c.append(colors[int(x[2])])
	else:
		incorrect_x.append(x[0])
		incorrect_y.append(x[1])
		incorrect_c.append(colors[int(x[2])])

plt.scatter(correct_x,correct_y,color=correct_c,marker = '+',s=75)
plt.scatter(incorrect_x,incorrect_y,color=incorrect_c,marker = '_',s=75)
f.close()
plt.show()

filename="errors.txt"
f = open(filename)

xs=[]

for line in f.readlines():
	xs.append(float(line))

plt.plot(xs)
f.close()
plt.show()