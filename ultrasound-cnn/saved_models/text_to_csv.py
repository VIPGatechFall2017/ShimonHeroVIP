import sys
filename = sys.argv[1]
with open(filename) as textfile:
    with open(filename[:filename.index('.')] + '.csv', 'w') as outfile:
        for line in textfile:
            if line.startswith('Epoch'): continue
            accpos = line.index('acc: ') + len('acc: ')
            acc = line[accpos:accpos+6]
            valaccpos = line.index('val_acc: ') + len('val_acc: ')
            val_acc = line[valaccpos:valaccpos+6]
            outfile.write(str(acc) + ',' + str(val_acc) + '\n')