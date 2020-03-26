import sys
import pandas as pd
import numpy as np
# from google.colab import drive 
#!pip install xlutils
# !gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output data.zip
# !unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

# x = np.empty([12 * 475, 18 * 5], dtype = float)
# y = np.empty([12 * 475, 1], dtype = float)
# for month in range(12):
#     for day in range(20):
#         for hour in range(24):
#             if day == 19 and hour > 18:
#                 continue
#             x[month * 475 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 5].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
#             y[month * 475 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 5] #value

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
#delete feature
# ith_feature = 0
# x = np.delete(x, [(9*ith_feature + i) for i in range(9)], axis = 1)


# x = np.delete(x, [i for i in range(9*9)], axis=1)
# x = np.delete(x, [i+9 for i in range(8*9)], axis=1)


a = [0,4,8,10,11,14,16]
x = np.delete(x, [i*9+j for i in a for j in range(9)], axis=1)
print( [i*9+j for i in a for j in range(9)])

import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

# import xlwt # 貌似不支持excel2007的xlsx格式
# from xlrd import open_workbook
# from xlutils.copy import copy
# # workbook = xlwt.Workbook()                   #create an excel
# # worksheet = workbook.add_sheet('test1123')
# rb = open_workbook('data.xls')
# rs = rb.sheet_by_index(0)
# workbook = copy(rb)
# worksheet = workbook.get_sheet(0)



# Training
trainset_size = math.floor(len(x) * 0.8)
dim = len(x[0])+1
# dim = 18 * 5 + 1
# worksheet.write(0,7,5)
w = np.zeros([dim, 1])
x_train_set = np.concatenate((np.ones([trainset_size, 1]), x_train_set), axis = 1).astype(float)
learning_rate = 5
iter_time = 10001
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
# counter = 1
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/trainset_size)#rmse
    if(t%1000==0):
        # worksheet.write(counter,6,t)
        # worksheet.write(counter,7,loss)
        # counter += 1
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
# workbook.save('data.xls')

x_validation = np.concatenate((np.ones([12*471 - trainset_size, 1]), x_validation), axis = 1).astype(float)
validation_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/(12*471 - trainset_size))
# x_validation = np.concatenate((np.ones([12*475 - trainset_size, 1]), x_validation), axis = 1).astype(float)
# validation_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/(12*475 - trainset_size))
print(validation_loss)


# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv(sys.argv[1], header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
# test_x = np.empty([240, 18*5], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]

test_x = np.delete(test_x, [i*9+j for i in a for j in range(9)], axis=1)
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

import csv
with open(sys.argv[2], mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)