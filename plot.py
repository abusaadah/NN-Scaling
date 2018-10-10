import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(10,100,20)

a = [21.84000015258789, 36.820000410079956, 92.17000007629395, 96.25999927520752, 96.20000123977661, 
	96.23000025749207, 96.24999761581421, 96.23000025749207, 96.31999731063843, 96.34000062942505, 
	96.29999995231628, 96.28000259399414, 96.31999731063843, 96.28000259399414, 96.31999731063843, 
	96.29999995231628, 96.27000093460083, 96.27000093460083, 96.28999829292297, 96.28000259399414] 

#a1 = [ 16.2,37.0,90.3,93.2,95.4,95.9,96.4,96.8,96.7,96.7,96.7,96.7,96.7,96.7,96.7,96.7,96.7,96.7,96.7,96.7]
a1 = [19,33,90,93,95,96,97,97,97,97,97,97,97,97,97,97,97,97,97,97]

plt.plot(x,a, 'g-^', label = 'First Profile')
plt.plot(x, a1, 'b->', label = 'Second Profile')
plt.xlim(0, 100)
plt.xlabel('IDP (%)')
plt.ylabel('Classification Accuracy (%)')
plt.title('Multiple Profile (MNIST)')
plt.legend(loc = 'upper right')
plt.ylim(93, 100, 1)
plt.grid()
plt.show()
