input_file = open('./raw_data.csv', 'r')
output_file = open('./raw_data_filtered.csv', 'w')
for line in input_file:
    output_file.write(bytes(line, 'utf-8').decode('utf-8', 'igonre'))
input_file.close()
output_file.close()

