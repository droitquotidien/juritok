import csv

for file in ['jorf_2023', 'jorf_2022', 'jorf_2021', 'jorf_2020']:
    with open(file + '.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        sentences = []
        for row in data:
            sentences.append(row[5])

    filePath = file + '.txt'

    with open(filePath, 'w', encoding='utf-8') as txtFile:
        for sentence in sentences:
            txtFile.write(sentence + '\n')
