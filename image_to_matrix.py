from PIL import Image as im
import numpy as np


i = 0
while i < 1741:
    if i == 0:
        matrix = im.open('zach_cont_gest_1/hand.mov_' + str(i) + '.png').convert('RGB')
    else:
        matrix = np.concatenate((matrix, im.open('zach_cont_gest_1/hand.mov_' + str(i) + '.png').convert('RGB')), axis=0)
    i += 1
    print('i = ' + str(i))

np.save('processed_cont_gest_1/processed_hand_matrix', matrix)