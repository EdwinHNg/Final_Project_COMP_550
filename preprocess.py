import csv
import re

def get_data(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        lines = []
        for row in reader:
            lines.append([row[0], process_text(row[1])])
    return lines

def process_text(text):
    # only include letters from the English alphabet
    res = re.sub('[^a-zA-Z ]+', '', text)

    # remove links
    res = ' '.join([s for s in res.split(' ') if 'http' not in s])

    # remove long words
    res = ' '.join([s for s in res.split(' ') if len(s) < 20])

    # remove multiple spaces
    res = re.sub(' +', ' ', res);

    # lowercase the entire string
    return res.lower()

def write_data(filename, data):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data:
            if row[1]:
                writer.writerow(row)

if __name__ == '__main__':
    data = get_data('tweets2.csv')
    write_data('tweets3.csv', data)
