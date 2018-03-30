# coding=utf-8
import csv
with open('face_attr.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    with open('face_attr.txt', 'rb') as filein:
        for line in filein:
            line_list = line.decode('utf-8').strip().strip('\n').strip('\r').split(' ')
            line_list = list(filter(lambda x : x != '', line_list))
            spamwriter.writerow(line_list)