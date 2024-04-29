from scipy.stats import kendalltau
import numpy as np
a = [18.23, 19.08, 20.72, 19.99, 19.69, 20.17, 18.75, 21.22, 20.71, 25.16]
b = [20.14, 21.46, 22.21, 21.38, 21.09, 21.65, 20.74, 22.6, 21.23, 25.34]
# a = [12,2,1,12,2]
# b = [1,4,7,1,0]
Lens = len(a)
 
ties_onlyin_x = 0
ties_onlyin_y = 0
con_pair = 0
dis_pair = 0
for i in range(Lens-1):
    for j in range(i+1,Lens):
        test_tying_x = np.sign(a[i] - a[j])
        test_tying_y = np.sign(b[i] - b[j])
        panduan =test_tying_x * test_tying_y
        if panduan == 1:
            con_pair +=1
        elif panduan == -1:
            dis_pair +=1
 
        if test_tying_y ==0 and test_tying_x != 0:
            ties_onlyin_y += 1
        elif test_tying_x == 0 and test_tying_y !=0:
            ties_onlyin_x += 1
 
Kendallta1 = (con_pair - dis_pair)/np.sqrt((con_pair + dis_pair + ties_onlyin_x)*(dis_pair +con_pair + ties_onlyin_y))
Kendallta2,p_value = kendalltau(a,b)
 
print(Kendallta1)
print(Kendallta2)
