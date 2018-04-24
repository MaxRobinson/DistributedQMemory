import numpy as np

import matplotlib.pyplot as plt


# one of 4 agents
# y = [-308, -875, -236, -371, -596, -722, -245, -371, -812, -866, -371, -551, -569, -236, -992, -254, -245, -506, -299, -542, -308, -281, -641, -344, -218, -317, -794, -488, -245, -236, -227, -290, -236, -450, -254, -641, -389, -1010, -452, -317, -245, -281, -1253, -290, -281, -389, -281, -236, -263, -245, -317, -254, -272, -254, -857, -272, -290, -695, -299, -929, -218, -1208, -272, -299, -299, -1019, -389, -353, -15, -281, -398, -272, -263, -245, -299, -299, -254, -569, -245, -326, -209, -272, -416, -353, -326, -245, -335, -254, -263, -668, -281, -245, -183, -470, -263, -308, -281, -263, -254, -272, -245, -263, -659, -398, -470, -112, -326, -245, -263, -281, -227, -299, -650, -245, -281, -254, -245, -254, -398, -236, -272, -263, -281, -461, -254, -254, -254, -227, -308, -33, -245, -227, -236, -272, -281, -245, -218, -245, -254, -236, -227, -389, -326, -227, -245, -218, -245, -569, -227, -195, -245, -236, -176, -254, -290, -299, -136, -272, -236, -902, -434, -206, -236, -99, -99, -24, -15, -254, -263, -139, -51, -263, -261, -254, -290, -24, -23, -69, -236, -245, -122, -172, -44, -245, -26, -71, -39, -8, -110, -509, -126, -53, -281, -176, -227, 9, -533, -236, -31, -20, -272, -263, -272, -236, -95, -245, -290, -49, -29, -234, -902, -713, -236, -27, 9, -37, -8, -281, -236, -35, -13, -13, -1, -146, -227, -245, -434, -263, -211, -4, -16, -299, 1, -245, -245, -272, -76, -18, -299, -236, -236, -130, 3, -39, -63, -7, -227, -272, -245, -299, -12, -12, -218, -33, -245, -36, -29, -133, -254, -173, -127, -227, -47, -245, -69, -29, -254, -1, -254, -20, -209, -31, -15, -236, -29, 6, -9, -122, -40, -254, -33, -6, -227, -46, 6, -212, -1, -23, -19, -3, -35, -169, 1, -2, -254, -19, -69, -29, -218, 6, -218, 0, -236, -227, 2, -25, -38, 8, -218, -236, -263, -85, -15, -66, -5, -14, -17, -21, 10, -272, -254, -209, -10, -54, -51, -6, -93, -20, -2, 4, 4, -254, -227, -13, -245, -61, -200, 3, 1, -136, -28, -47, 1, -48, -7, 9, -7, 9, -41, -82, -153, -65, -17, -118, -40, 12, -2, 5, -23, -227, 13, -8, 8, 5, -236, -72, -18, -12, 4, -24, 11, -8, -227, -25, -4, -5, 13, -263, -4, -8, -33, 4, -254, 2, -245, -45, -2, -213, -13, -149, 13, -22, 7, -43, 8, 6, -80, 14, -7, -10, -14, -15, -263, 5, -29, -263, -6, 8, -73, -3, -83, 8, 8, -32, 2, -26, 4, -40, 12, -56, 6, 0, -34, -22, -18, -2, -6, -1, 10, -12, 8, -3, -12, 7, -25, -140, -41, -54, 4, -16, 1, 2, -49, 7, 7, -1, -38, -90, 7, -11, 10, -56, -33, -46, -76, 4, -69, 11, 7, -54, -64, 7, -8, -17, 9, -10, 1, -2, 10, 4, -5, -14, -26, 1, 10, 15, -58, -22, -3, -11, -21, -3, 2, -7, -11, -50, 8, -74, 0, -14, 10, -5, -16, -16, 11, -38, 4, -14, 8, 13, -45, -281, -10, -16, -38, -20, 11, 9, 7, 10, -23, -20, 7, 5, 10, 10, 6, -33, 7, -4, -2, -9, -77, 10, 2, -23, 10, 2, -78, 9, 8, -10, 3, 8, -16, 12, 5, 8, 5, -4, -59, -133, -2, -19, 8, -25, -6, -2, 7, 10, 11, -3, 10, 4, 8, -17, -12, -67, -4, -51, 9, 5, -6, -5, -74, 5, 0, -30, -109, 3, 7, -9, 7, 5, -8, -45, 4, 3, 0, 8, -10, -4, 5, -11, 6, 8, -1, 0, -4, 11, -11, 11, 11, 2, 7, 7, -25, 10, -105, 8, 9, -1, 4, 4, -56, -16, 9, 9, 6, -21, 12, 7, 5, 9, -7, -12, 9, -7, 1, -4, -13, 14, 12, -10, 9, -5, 1, 7, -43, -10, -14, -76, 5, 9, -12, 8, -40, 8, 6, -12, 9, 3, 0, 9, 9, -34, 6, 2, 10, -4, -58, 0, -18, -49, -5, 8, -4, -8, 6, -9, 11, -30, 5, -47, -13, 3, -93, 4, 2, -8, -8, 7, -1, 7, 4, -9, -10, 8, 2, 11, -2, 3, 5, 7, 7, 5, 5, 2, 4, -9, -3, 7, 8, -30, -12, 10, 15, -6, 5, -3, 7, 7, 9, 6, 10, -6, -2, -2, 7, -41, 3, 7, -2, 3, 10, 13, 8, 6, -10, -7, 0, 6, -5, -46, -15, 11, 6, 1, 9, 11, 4, -46, 2, 11, -16, 12, -3, 11, -14, -31, 4, -9, -6, -9, 9, -24, -9, 4, -2, -40, -6, 2, 7, 6, 0, 6, -104, 7, -1, 13, 10, -2, 4, -21, 9, 0, 5, -7, 2, 9, 11, -11, 12, -7, -5, -1, 5, 2, -9, 8, 10, 8, -6, 11, 5, 15, 2, 5, 5, 7, -6, -10, 4, 0, 0, 5, -3, 10, -20, 0, 12, -2, 8, 2, 2, -2, -1, 6, -12, 3, 4, 5, 4, -4, 10, 8, -3, 4, 8, 1, 8, -21, -2, 10, 7, 15, 10, -1, 6, 9, 9, 9, 4, 3, 11, 10, -2, 2, -29, 1, 1, 7, -6, -3, 6, 8, 5, 0, -8, 10, 13, 1, -13, -43, 1, 8, 12, 9, 8, 8, 9, 4, 11, 9, 6, 9, -93, 4, -3, 12, -26, 2, 3, 0, 12, 10, 9, 3, 6, 0, 7, -12, 7, 11, -7, -16, -19, 9, 0, 10, 5, 9, 8, 8, -63, 5, 8, -5, 9, -11, -3, 10, 11, 4, 1, -10, -5, -3, -7, 6, 14, 6, 2, 0, 8, 10, 1, 13, 6, -2, -40, 10, -1, 10, 5, 7, 3, 5, 7, 11, 9, 8, 11, 10, 8, 2, 6, 11, 5, -8, 14, -4, 11, 6, 4, 5, 5, 0, 8, 9, 10, 4, 6, 4, 3, 3, 1, 10, 12, 0, -16, 9, 14, 9, -1, 4, -7, 6, -28, 13, 6, -3, 7, 5, 11, 4, 6, 7, 3, 2, 1, 10, -13, 12, 10, 2, -2, 4, 0, 4, -5, -9, -12, 7, 2, 6, 10, -20, 4, -7, 10, 1, 9, -9, 0, -8, 9, 7, 9, 1, 5, 12, 7, 14, -6, -1, -11, 1, 9, 4, 7, 8, -24, 2, 9, 2, 9, 9, -8, -6, 5, 11, 13, 9, -4, 6, 4, 4, -19, -10, 7, 0, 5, 3, 6, 3, -8, 0, 6, 9, 4, 0, -9, 12, -3, 0, -2, 3, 4, -8, 3, 8, 7, 6, -1, 7, 4, 14, 1, 5, 1, 0, 0, 7, 4, 8, -5, 9, -14, -1, -6, 11, -18, 4, 13, 11, 10, 9, 9, 5, 6, 3, 7, 3, 4, -11, 7, 3, 10, 6, 2, 5, 7, 9, -9, 6, 4, 11, -3, 8, 8, -5, 8, -7, 5, 1, 12, 8, 9, -53, 6, 12, -7, 7, 10, 11, -5, 3, 5, 7, -6, 9, 4, 8, 11, 14, 1, 5, 9, 8, 10, 5, 10, 5, 7, 5, 9, 10, 9, 12, -5, 11, 10, 9, 14, -28, 5, 1, 11, -1, -15, 10, 12, -12, 15, -1, 7, 12, 9, 10, 1, 10, 8, 11, 12, -5, 3, 10, -9, 12, 10, 2, 4, 9, 10, 12, -11, -5, -7, 12, 12, 9, 5, -8, 6, 10, 1, 11, 7, 5, 11, 3, 9, 2, 0, -14, -5, 7, -3, 12, 13, 8, 5, 8, 6, 6, -4, -1, 3, 10, 0, 0, 1, -7, -5, 5, 7, 8, 10, 3, 4, 5, 0, 7, 12, -1, 10, 4, -11, 0, 12, 5, 7, 6, 0, 6, 3, 1, 10, -2, 14, 5, 6, 10, 11, 2, 9, 15, 8, 11, 8, 2, 6, 0, 10, 6, -2, 7, 4, 7, 5, 4, 11, 7, 9, 0, -1, 5, 2, 4, 9, 4, 0, 8, 6, 1, 6, 11, 9, 5, 8, 4, 4, 2, 12, 8, 9, -3, 3, -1, 8, 6, 8, -1, 6, 7, 13, 12, -22, 1, 9, 10, 4, 10, 7, 7, 3, 7, 4, 4, 7, 6, 7, 4, 7, 7, 7, 14, 0, 7, 10, 2, 13, 8, -5, -13, -6, 9, 3, -1, 0, 14, 11, 6, 7, 10, 11, 1, -1, 7, 10, 3, 8, 6, 8, -5, 7, 10, -2, 12, -2, 7, 8, 2, -1, 1, 6, -8, 8, 8, 9, 6, 9, 7, 1, 10, 5, 6, 5, 10, -1, 7, 2, 3, 4, 0, 2, 4, 4, 2, -3, 7, 2, 2, 2, 6, 6, 10, 8, 11, 15, 3, -3, 6, 8, 8, 12, 0, 1, 8, 3, 13, 10, 8, 9, 10, 5, 7, 4, 7, 11, 8, 7, 3, -1, -9, 9, -8, -4, 2, 13, 4, 7, 8, -3, 8, 9, 13, 9, 4, 7, 8, 2, 10, -4, 5, 8, 2, 4, 8, 11, 2, 8, 2, 1, 13, 2, 6, 9, 4, 5, -3, 5, 6, 10, -5, 3, 9, 10, -15, 11, 9, 8, 0, 8, 11, 8, 2, 3, 9, 11, 6, 10, 3, 2, 7, 10, 6, 7, 8, -6, 5, 3, 7, 7, 5, 8, 3, 8, 9, 11, 2, -6, 3, 10, -2, 7, 3, 11, 6, 5, 7, 8, 10, 5, 6, 14, 7, -5, 14, 10, 1, 6, -8, 10, 10, 8, 4, 7, 8, 5, 6, 10, 8, 6, 9, 2, 8, 6, 9, 12, 10, 3, 8, 0, -1, 13, 7, -6, 12, -15, 10, 9, 8, 8, 3, 5, -1, 5, 1, 6, -19, 11, 7, 4, 12, 6, 10, 7, 5, -1, 9, 6, 0, 8, 7, 0, 1, 5, -3, 1, 4, 7, 3, 11, 10, 4, 4, 9, 12, 11, 4, 3, 7, 9, 10, 6, 7, 8, 7, -1, 8, 7, 5, 4, 4, 10, 1, 1, 11, -2, 10, 10, 4, 7, 1, 10, 6, 1, 7, 12, 13, 10, 2, 13, 11, 5, -7, -21, 10, -1, 12, 7, 6, 2, 8, 7, 12, 9, 7, 7, 6, 6, 7, 4, 12, 9, 1, 3, 5, 1, 8, 10, 5, 2, 7, 6, 5, 8, 3, 8, -1, 8, 7, 1, 10, 4, 11, 0, 10, 7, 10, 10, 6, 7, 13, 9, 6, 7, 11, 7, 5, 4, 7, 6, 8, 6, 10, 8, 6, 11, 8, 12, 9, 9, 7, 7, 7, 7, 8, 4, 10, 4, 5, 14, 7, 0, 7, 9, 6, 7, 5, 5, 9, 7, 8, 5, 8, 5, 8, 15, 7, 9, 4, 4, 10, 8, 9, 8, 10, 6, 7, 5, 4, 0, 3, -3, 12, 8, 10, 11, 6, 1, 8, 7, 9, 3, 6, 11, 3, 4, 9, 3, 5, 6, 9, 5, 10, 7, 9, 3, 12, 11, 0, 6, 5, 5, 5, 11, 4, 10, 2, -27, 9, 7, 2, 6, 7, 11, 9, 6, 7, 10, 6, 12, 10, 14, 8, 5, 2, -7, 10, 8, 8, -3, 8, 6, 12, 12, -2, 3, 4, 7, 8, 9, 11, 8, 4, 8, 8, 9, 3, 10, -5, 12, 10, 3, 9, 9, 7, 5, 8, 10, 14, 11, 10, 6, 11, 6, -1, 6, 11, 13, 6, 13, 10, 5, 6, 10, -3, 5, 5, 7, 7, 2, -3, 8, 10, -2, 12, 7, 7, 4, 9, 9, 7, 9, 4, -1, 7, 14, 10, 5, 11, 6, 9, 9, 8, 4, 9, 4, 11, 8, 9, 5, 7, 4, 4, 10, 6, 2, 6, 5, 12, 7, 9, -1, 11, 3, 8, 10, 7, 4, 6, 10, 7, 7, 13, 6, 9, 7, 3, -3, 7, 2, 5, 2, 8, 7, 8, 4, 14, 7, 3, 13, 3, 3, 7, 9, 1, 1, -4, 4, -18, 7, 5, 7, 12, 3, 8, 9, 8, 11, 5, 12, 9, 11, 5, 7, 9, -1, 8, 8, 1, 7, 6, 5, 6, 7, 1, 5, 2, 7, 7, 10, 1, 6, 5, 10, 8, 8, 7, 9, 2, -1, 5, 0, 6, 9, 6, -2, 7, 7, 4, -3, 8, 7, 5, 8, 8, 10, 6, -10, 7, -3, 7, 9, 10, 10, 15, 6, 11, 5, 5, 7, -3, 10, 9, 8, 12, 10]

# 1 agent
# y = [-956, -668, -659, -533, -317, -650, -821, -965, -749, -740, -407, -281, -578, -920, -461, -578, -713, -263, -434, -299, -425, -452, -632, -438, -416, -830, -515, -227, -308, -245, -344, -461, -587, -344, -641, -362, -731, -461, -488, -272, -227, -668, -272, -254, -263, -245, -362, -236, -218, -281, -434, -254, -461, -272, -623, -272, -236, -245, -227, -236, -245, -497, -236, -254, -299, -263, -254, -236, -236, -416, -263, -245, -281, -443, -72, -272, -245, -23, -245, -254, -326, -272, -227, -236, -776, -389, -569, -236, -272, -362, -452, -236, -263, -281, -497, -263, -14, -263, -245, -227, -650, -308, -254, -281, -281, -227, -254, -254, -254, -15, -290, -236, -263, -263, -281, 13, 13, -31, -254, -245, -254, -245, -299, -425, -263, -254, -245, -281, -254, -97, -632, -272, -281, -13, -48, -27, -245, -245, -281, -290, -407, -254, -207, -61, -344, -254, -3, -18, -263, -133, -326, -227, -49, -236, -254, -695, -263, -290, -362, -272, -19, -5, -2, -299, 13, -254, -209, -353, -245, -299, -263, -245, -272, -272, -13, -254, -87, -236, -254, -20, -218, -63, -272, -272, -263, -236, -263, -290, -263, -9, -506, -245, -281, -767, -245, -263, -1127, -263, -254, -281, -227, -290, -227, -60, -44, -236, -236, -290, -758, -245, -245, -245, -209, -218, -245, -263, -127, -281, -236, -236, -227, -227, -245, -96, -290, -44, -218, -218, -272, 4, -26, -398, -272, -68, 10, -7, -299, -236, -236, -245, -263, -57, -236, -254, -290, -227, -31, -272, -245, -236, -263, -218, -32, -245, -254, -86, -91, -227, -114, -254, -71, -254, -245, -35, -236, -32, -227, -247, -108, -227, -263, -88, -254, -281, -218, -272, -245, -272, -227, -209, -209, -31, 5, -80, -272, -236, -272, -272, -20, -227, -281, -218, 6, -470, 1, -263, -245, 9, -10, -218, 8, -245, -254, -30, -272, 11, -227, -281, -290, -245, -272, -272, -245, -12, 3, -227, -218, -254, -245, -236, -254, -227, -227, -245, -245, -26, -263, -263, -245, 14, -69, 0, -64, -66, -254, -263, -272, -227, -218, -272, -272, -263, -254, -236, -184, -68, 5, -209, 3, -245, 2, -56, -254, -29, -227, -254, -245, -281, -11, -245, -3, -299, -12, -227, -218, -9, -16, -245, -272, -47, -227, -227, -263, -245, -254, -58, -236, -26, -218, -209, -28, -227, -95, -236, -209, -254, -236, -254, 8, -263, -236, -236, -245, -254, -263, -254, -218, -236, -263, -209, 1, -245, 4, -254, -67, -227, 0, -254, -272, -209, 6, 9, -263, -121, -13, -227, -227, 6, -19, -218, -272, 6, -218, -245, -5, -263, -263, -254, -254, -236, -263, -245, -9, -2, -218, -33, -59, -236, 11, -2, 8, -35, -227, -14, -2, -272, 9, -15, -263, -272, -227, -2, -236, -218, -245, -245, -25, -236, -263, 11, -272, 4, -218, -227, 1, -227, 8, -26, -15, -227, -227, -236, -263, -254, -227, -236, -218, -263, -9, -4, 9, -12, 10, -236, -218, -245, -254, -6, 13, -236, -263, -227, -3, -227, -227, -236, -245, -245, -254, -209, -254, -227, -263, -254, -236, 1, 7, -254, -31, -227, 9, -1, -227, -254, -42, 7, 3, -5, -236, -4, -227, 11, -209, -227, -10, -263, -227, -3, -209, -227, -2, -227, -19, 4, -227, -10, -30, 8, -15, -218, -116, -227, -227, -254, -227, 9, 7, -2, 0, -209, -11, -1, -218, -254, -236, -263, -236, 2, -10, -227, -245, -272, 4, 7, -245, 3, -245, -236, 13, 3, -236, -1, -227, -236, -1, -245, -236, -263, -236, -227, -218, -227, -227, -218, -218, -245, -227, -263, 13, -245, -16, -236, -245, 5, 2, -227, -6, -3, -236, -3, -227, -5, -281, 4, 7, 3, -209, -227, -218, -218, -2, -227, -2, -272, -11, -236, -272, -245, 10, 0, -209, -272, -236, 3, -218, 10, -254, -209, -218, -218, 10, -15, -236, 11, -236, -8, -218, 2, -236, -254, -13, -281, -12, -209, -227, -209, -236, -227, -236, -227, -227, -218, -7, -263, -218, -227, -254, 2, -236, 8, 0, -209, -245, 0, -18, 1, -227, -227, -227, 3, -263, -218, 6, -8, -245, -272, -180, -2, -227, -227, -254, -8, 1, -236, -166, -2, -227, -227, 6, -171, 10, -245, 4, -227, -227, 7, -1, -183, 5, 7, -4, -218, -11, 12, 5, -227, -2, -245, -236, -135, 11, -3, 8, 7, 10, -15, -254, -118, -263, -12, 1, 10, 8, -227, -212, -37, -8, 12, 2, -227, -1, -227, -218, -227, -209, -209, -209, -6, 5, 5, 7, 1, 1, -200, -218, -254, 4, -2, 3, 6, -4, 6, 7, -34, -173, 10, -227, 1, -16, -218, 11, -4, 3, 13, -4, -1, -227, 5, 11, -245, -128, -209, -227, 6, 7, -227, -263, -236, -123, -236, 8, -218, -209, -8, -3, -81, -209, -6, -227, -2, -227, -8, -90, -218, -236, -7, 8, -227, -209, -209, 4, -227, -200, -3, -152, 4, -200, 10, -1, -209, -245, -7, -113, -218, 8, -39, -254, 5, -236, -28, 4, 5, -3, -2, -209, -161, -227, -218, 1, 9, -28, -23, -33, -227, -86, -236, -2, 1, -28, -227, -227, -218, -15, -209, -227, 3, 8, -245, -7, -245, -245, -218, -73, 11, -32, -227, -6, 2, -97, 9, 6, -12, -245, -4, -47, 8, -44, 5, -80, -57, -218, -218, -105, -53, -209, -14, -18, -1, -102, 6, 7, -227, -236, -5, 2, -101, -13, -218, -2, -13, 7, -41, -99, 6, -35, 9, -120, 11, 6, -227, 15, -22, -120, -28, -106, 4, -161, -1, -17, 4, -115, -42, -38, 3, 9, 10, -218, -29, -7, -7, 14, 6, -93, 4, -136, 1, -209, 1, -159, 0, 3, 10, 12, -5, 9, 5, -17, 14, 4, -87, -99, 12, 8, -53, 0, -15, 13, 6, 6, -165, -113, -140, 11, 14, 8, 9, 11, 7, -19, -170, 5, -117, 6, 6, -19, -27, 6, -33, -13, 3, -190, -3, -57, -178, -97, -4, 10, 10, -1, 7, -12, -124, 9, 5, 11, -10, -14, 10, 5, -2, -71, -14, 7, -8, 13, -5, 7, -254, -17, -130, 6, -188, 3, -149, -16, -51, 9, 11, -45, -95, 9, 3, 12, -3, -227, -91, -5, 8, 10, 5, -2, -5, 4, -55, 11, 9, 6, -13, 4, -65, 0, 3, -15, -93, -62, 8, 3, -16, 7, -5, -64, 8, 11, -2, 11, 6, -56, -19, -1, 9, -3, -72, -13, 10, -107, 10, -9, 11, 11, -79, 6, -12, 7, 5, 6, -92, -17, 0, -131, -153, 9, 14, 6, 15, -73, 2, -1, 4, 9, 0, -14, 6, -6, -34, 11, 8, -12, 8, 9, -13, 11, -1, 1, 4, -76, -9, 5, 6, 11, -116, 5, -10, 8, 12, -30, 13, 10, 7, 12, -38, -4, -5, 13, 9, 7, -25, 0, -2, 10, 7, 7, -83, 11, -23, 11, 3, -6, -53, -47, -169, -2, 6, 4, 8, -3, 10, -31, 9, -6, -54, 3, 4, 9, -150, -10, 3, -44, 1, 6, 10, 10, -49, 6, 11, 3, 7, -23, -4, 8, 11, -46, -25, 11, 7, 8, -5, 1, 8, 8, 5, 10, -95, 12, -6, -10, -4, -6, 7, 8, 12, -57, -3, 11, 5, 11, 9, -4, -34, 6, 4, 5, 6, -2, 9, -3, 4, 5, 10, 9, -6, 11, 11, 9, 2, 6, 6, 8, 6, 9, -4, -22, 5, 8, 5, 7, 6, -6, -37, 9, 5, -8, -4, 3, -5, 5, 5, 0, -6, -9, -19, 2, 8, 13, -16, -15, -46, 11, -3, -30, 4, -57, 9, -37, 11, 11, 0, -6, 6, 12, 6, 4, -89, -19, 4, 12, -43, 7, -4, 4, 10, -62, 3, 6, -22, 9, -3, -13, 1, 11, 6, 9, -39, 0, 9, 9, -36, 11, -9, 8, 5, 3, 11, 4, 3, 0, -57, 5, -8, -11, 8, 7, 5, -6, 7, 9, -3, 1, 12, -35, 7, 6, 7, 5, 3, -3, 6, 8, 10, 8, 5, -7, 6, 6, 0, -1, 0, 7, -4, 10, 7, 8, 3, 9, 3, 6, 5, -10, 7, -8, -62, 4, -12, 4, 7, 14, 9, -4, 11, -4, -21, 4, 12, 9, 8, -4, 12, -6, 11, -1, -16, 12, 1, 14, -25, -5, 5, -17, 6, 9, -1, 10, 11, 11, 11, -17, 10, -9, -3, 10, 3, 6, -7, -20, 10, -9, 13, 7, 14, 4, 0, 5, 7, -21, 2, 0, 11, 7, 11, 9, 10, 7, 4, -1, 11, 10, 1, 11, 8, 9, 11, 3, 7, -3, -22, 8, -14, 1, 6, 9, 5, 7, 6, -18, -40, 7, 3, 5, 1, 1, 3, 11, -3, -6, 6, 14, 7, 13, 11, 15, 0, -6, -23, 2, 4, 6, 5, 4, 5, -20, 7, -41, 10, 7, -16, 3, -1, 6, 13, 4, 7, -21, -26, 6, -43, 7, -1, 7, 12, 7, 5, 9, 7, 9, -24, 7, -13, 3, 5, 4, 10, -10, 8, 5, 9, 10, 7, -24, 6, 9, 1, 5, -14, -6, 8, 6, 11, 8, -8, 3, 11, 6, 6, 11, 1, 12, 5, 7, 9, 5, 3, 7, 6, 8, -25, 11, 4, 5, -2, 8, 7, -3, 0, -22, 4, -1, 6, 5, 11, 11, 6, -6, 7, -6, 7, 11, 10, 6, 8, 15, -4, -11, 10, 8, 3, -5, 8, 9, 5, 1, 6, 9, -12, 10, 12, 3, 9, 14, 4, 3, -23, -5, 6, -3, -28, 1, 7, 9, 1, 6, 5, 2, 5, 6, 6, 7, 10, 12, 7, 9, 10, 5, 6, -1, 8, 13, 9, 7, 2, 7, 8, 8, 8, -3, 13, 9, 8, 13, 10, -1, 10, 5, -16, 10, -21, 5, 6, 3, 6, -1, 11, 8, 5, -8, 11, 8, -11, -1, 3, -5, 7, 9, -16, 11, 5, 5, 10, 8, 7, 6, 6, 2, 11, 7, 11, 6, 8, 7, 12, 8, 12, 9, -4, -15, 9, 5, 10, -39, 8, 3, -12, 4, -3, 6, 8, 11, 3, 3, 11, 11, 10, 8, 2, 6, 6, 14, 9, 6, 0, 12, 7, 7, 11, 11, 9, -1, 9, 9, 9, 15, 2, 9, 11, 2, 1, 5, 10, 5, 8, -5, -8, 13, 4, 5, 8, 6, 11, 7, -5, 11, 9, 7, 7, 10, -7, 14, 3, -4, 3, 5, 2, 12, 11, 7, 3, 9, 9, 15, -1, 10, 7, 13, 8, 12, 6, 6, 8, 7, 11, 6, -4, -5, 9, 7, 9, 4, 15, 5, 8, 6, 9, 9, 14, 6, 9, 12, 12, 5, 14, 6, 11, -1, 6, 6, 5, 4, 3, 8, -4, -2, 3, -5, -3, 0, 6, 9, 5, 4, 3, 12, 9, -12, 4, 5, 8, 11, 4, 7, 7, 8, 7, 3, 10, 5, 11, -1, 10, 3, 4, 11, 7, 5, -1, 11, 9, 9, 5, 4, 0, -4, 7, 5, 8, 6, -2, 13, 6, 6, 8, 5, 3, 5, -1, 1, 5, 11, 6, 5, 6, 7, -3, 3, 4, 0, 12, 9, 8, -1, 5, 9, 10, 13, 11, 5, 11, 4, 4, 4, 10, 9, 7, 9, 9, 14, 3, 5, 6, 3, 7, -3, 9, 11, 7, 9, 4, 5, 7, 4, -3, 14, 7, 13, 4, 8, 14, 15, 6, 12, 8, 10, 9, 10, 11, 10, 11, 11, 10, 2, 8, 7, 9, 6, 12, 3, 14, 5, 5, 9, 5, 7, -14, 9, 9, 4, 7, 11, 9, 9, -1, 7, 2, 10, 8, 4, 3, 8, 8, 9, 11, 7, 9, 6, 11, 1, 3, 9, 5, 10, 6, 12, -2, 6, 6, 8, 5, 6, 7, 4, 7, 5, 0, 3, 4, 9, 7, 5, 5, 4, 10, -1, 8, 15, 3, 7, 12, 13, 11, 10, 10, 5, 6, 3, -3, 7, 4, 9, 6, 5, 6, 6, 6, 9, -6, 9, 4, -13, 9, 11, 5, 8, 6, 4, -6, 8, 10, 6, 4, 8, 6, 11, 3, 9, 13, 8, -4, 5, 6, 8, 13, 14, 8, 8, 9, 15, 11, 10, 8, 2, 11, -4, 6, 7, 7, 8, 9, 15, 10, 11, 6, 7, 7, 11]

# ref for 8 agents
# y = [-236, -281, -245, -236, -281, -254, -245, -290, -272, -290, 1, -263, -218, -245, -272, -254, -272, -254, -272, -191, -254, -236, -245, -23, 8, -2, -3, 11, -13, 12, -234, -39, 11, 7, -3, -12, 7, -184, 14, -53, -4, 4, -4, -227, -75, 9, 8, -8, -217, -112, 11, 9, -3, -3, 11, 12, 0, 11, -78, -3, 8, -5, 10, -84, 7, 7, -9, 0, 1, 7, 9, 8, 9, 10, 10, 4, 3, 12, -15, -1, 6, -3, 11, 6, 5, 4, -19, 5, 7, -2, -3, 9, 9, -58, 11, -10, -3, -3, 9, 9, 5, 13, 8, 5, 11, 7, -3, 6, -4, 7, -25, 4, 1, 11, 0, -2, 9, 6, 4, 7, 9, 8, 8, -4, 11, 7, -1, 10, -2, 10, 9, 8, -23, 5, -20, 13, -4, -2, 3, -3, 6, 14, 9, 5, 2, 14, 10, 5, -4, 5, 12, 10, 8, -3, 12, 1, 10, -13, 12, 10, 7, 10, -3, 10, 6, 12, 9, 7, 8, -8, -4, -6, 14, 8, 7, -3, 10, 8, 6, 14, 5, 11, 12, -3, -4, 9, -3, -14, -13, 6, 7, 1, 7, 11, -17, 7, 3, 8, 6, 8, 7, 12]

# one of 8 agents
y = [-398, -326, -272, -317, -740, -443, -911, -596, -245, -272, -236, -740, -254, -740, -1226, -983, -317, -317, -794, -326, -479, -362, -560, -569, -263, -218, -425, -866, -650, -263, -308, -335, -344, -263, -263, -203, -362, -263, -353, -254, -272, -479, -263, -245, -632, -227, -245, -380, -551, -263, -290, -299, -830, -317, -497, -317, -226, -245, -371, -704, -254, -272, -362, -245, -236, -218, -479, -236, -272, -308, -425, -254, -623, -272, 13, -281, -236, -479, -22, -62, -254, -263, -299, -236, -19, -245, -91, 1, -299, -236, -245, -20, -335, -254, -263, -443, -1109, -218, -245, -308, -272, -263, -353, -677, -299, -227, -67, -218, 3, -236, -866, -254, -281, -206, -236, -127, -72, -194, -227, 5, -151, -290, -127, 6, -8, -43, -281, -70, -245, -45, -37, -398, -171, -154, -497, -236, -290, -198, -25, -12, -245, -281, -317, -236, -254, -65, -44, -173, -39, -164, 0, -113, -11, 7, 11, -8, 7, -245, -281, -104, -150, -120, -235, -174, -39, -22, 14, -272, -293, -263, -142, -6, -14, -93, -9, -272, -4, -18, -142, -740, -254, -150, -44, -245, -380, -44, -18, -245, -60, 7, -56, -209, -10, -398, -245, -79, 11, -254, -110, 7, -103, -281, -3, 3, -254, -16, -18, -2, -281, -245, -21, -174, -30, -272, -263, -68, -62, -124, 9, 2, 10, -15, -192, -132, -48, 10, 7, 3, -146, -16, -272, -47, -3, -272, -2, -10, -47, -4, -11, 6, -26, -4, -29, 5, -73, 3, -3, -7, -81, -63, -308, 0, -236, -272, 1, -19, -25, -1, -8, 1, -263, 12, -17, -24, -254, 8, -272, -209, -236, -41, -34, -88, 13, -47, -18, -3, -12, 7, -8, -263, -1, -12, -21, -9, -67, 7, -76, -1, -12, 9, -2, -5, 13, 6, -623, 7, -6, 3, 4, -19, -3, -236, -50, -57, -1, 9, 10, 1, 6, -22, -32, -15, 5, -371, -91, -14, 1, -227, -208, -22, -48, -62, -27, -106, -14, 11, -20, -4, -97, 8, -4, 3, 3, 5, -11, 14, -34, -70, 9, 9, -69, 5, -4, -2, -83, -7, -18, -65, 10, -41, -144, -102, 12, -6, -10, -28, 13, -94, -54, 2, -11, -23, 10, -14, -30, -134, -3, 11, -2, -2, -55, 3, -31, -36, -6, -3, 11, 8, -10, 1, -4, -52, 7, 8, 10, -47, 10, 9, 8, -7, 4, 7, -9, 9, 7, -5, 6, -9, 5, -14, -20, -12, -35, 12, -59, 2, -152, -8, -5, -2, 7, -3, 9, -8, -59, -1, 6, 1, -8, 8, -5, -7, 2, -6, -69, 6, -19, -17, -21, 8, -14, -46, 5, -3, -3, -3, 11, -9, -16, 7, -123, 5, -6, 0, -7, -23, 1, 8, 3, -1, 10, -6, 6, 3, -2, -3, -15, 7, 7, 10, -3, 8, 6, 5, 8, -9, -6, 9, -4, 8, 12, 8, -3, -5, 9, 8, 7, 2, 8, 7, 1, 12, -93, 5, 10, 3, 5, 7, -63, -9, 3, 9, -10, -378, -1, -10, 5, 5, 4, -94, -4, -8, 0, 8, -17, 7, -5, 8, -3, 4, 10, 0, 7, 6, -65, 15, -6, 1, 2, -3, 12, 8, 11, 7, -4, 2, 10, -8, -7, -2, 1, 0, -3, 5, 8, 4, -9, 10, 10, -9, -13, 2, -3, 11, -42, 10, 7, -4, 3, 13, 4, -11, 9, 3, 11, 7, 7, 4, 7, 0, -5, 5, -28, -7, 10, -16, 3, 10, -14, 6, -6, 4, -1, 4, 11, 3, 4, 11, -6, 10, 11, -13, -2, -3, 11, -9, 5, -2, 1, 11, -2, -9, 8, 11, 5, -22, 2, -1, 2, -8, 5, 1, 4, -48, -8, -25, -9, 1, -6, -8, -5, 10, 2, -2, 12, 4, 10, 7, 4, 6, 5, 1, 2, 12, 3, 4, -1, -2, 8, -9, 6, 2, -2, 4, 2, 6, -69, 5, -1, -13, 4, 3, 11, 3, -14, -19, -11, 1, 11, 6, 3, -21, 11, -2, 8, -43, 7, 4, -4, -3, 6, 4, -17, 3, 5, 5, 4, -18, 8, 6, -142, -8, 7, -371, 12, 7, -2, 4, 6, 5, 2, -1, -27, 4, 7, 4, 6, 8, 2, 2, 13, 8, 3, 10, 9, 7, -7, -9, -3, 4, 5, 5, 11, 3, 0, -227, 2, 5, 10, -10, 6, 1, 5, 2, -2, 6, 5, 8, -2, 7, 1, 8, 10, 5, 0, 5, 4, 10, 7, 1, -8, 0, 13, 3, 1, 8, 11, 4, 7, -12, 7, 9, 7, 6, 8, 11, 1, 0, -7, 10, 3, 6, 0, -1, -8, 9, 10, 6, -14, 1, 10, 5, -5, 4, 5, 9, 6, 7, 4, 4, 5, 8, -12, 8, 1, 13, 9, 3, 4, 5, 9, -6, 4, 7, 10, 6, 6, 3, 0, 8, 6, 8, 8, 9, 3, -1, 8, 12, 14, 7, 8, 7, 6, -14, 12, 5, 0, -1, -7, 10, 4, -12, 3, 2, 7, 3, 6, -1, 15, 3, -200, -7, 8, -5, 1, -6, -6, 6, -38, 9, -5, 7, -1, -16, 3, 11, 13, 9, -17, 9, 5, -209, -7, 0, 9, 7, -10, -4, -3, 3, 11, 2, -6, 4, 7, 1, 9, -8, -6, 5, -14, 7, 6, -2, -7, -7, 2, 9, -6, 8, -1, 10, 12, 2, 9, -8, 7, 5, 7, 4, 10, 5, 9, 4, 11, 5, -2, 7, 8, 3, 9, 4, -6, 9, 0, 12, 5, 7, 6, 3, 4, 8, 9, 7, 9, 7, 12, 8, 0, 0, 8, 11, 10, 6, 8, -5, 0, -2, 7, 8, -7, 2, -4, -2, -5, -11, 4, 5, 6, -36, 8, 7, 8, 9, 6, 11, -33, 2, 8, 2, 0, 6, -6, 7, 13, 8, -8, -9, 5, 9, -3, 4, 3, 6, 7, 0, 1, 3, 8, 3, 13, 12, -14, 5, 12, -94, 9, 3, 2, -12, 12, 2, 5, 3, 6, 3, 1, 11, 6, -6, -14, 11, 1, 10, -170, -6, 8, 12, -2, 3, 9, 5, 10, 4, 7, -6, 11, -5, 9, 7, 7, -9, 7, 10, 4, 10, 8, -4, 6, 12, -5, 11, 8, 4, 5, 6, 7, -1, 4, 8, 5, -4, 1, 9, 11, 3, 13, -8, 7, 5, -5, 8, -5, 8, -2, 5, -2, 7, 8, 8, 9, 0, 6, 3, 7, 7, 10, 10, 10, 3, 2, 9, 11, 8, 10, -6, 3, 3, 7, 7, -5, 4, 9, 11, 5, 11, 3, 2, 8, 7, -7, 3, 9, 5, 8, 5, 13, 6, 5, 9, 8, 7, 0, 7, -2, 8, 4, 7, 6, 9, 3, 7, 4, 3, 8, 0, 1, -5, 13, 8, 5, 9, 10, 10, 8, 9, 11, 8, 3, 8, 7, 5, 4, 8, 6, 7, 3, 1, 11, 9, 1, 5, 9, 5, 8, 2, 10, 0, 9, 3, 5, 7, 6, 5, 9, 9, -3, 14, 6, -1, 5, 10, 3, 6, 9, 8, 9, 7, 5, 6, 7, 8, -7, 5, 6, 11, -6, 9, -8, 10, 4, -5, 8, 12, 8, 6, -7, 8, 6, 8, 12, 5, 10, -3, 7, 6, 6, 5, 2, 6, 5, 4, 10, 10, 12, 5, 3, 6, -16, 3, 6, 8, 6, 7, 6, 2, 7, 6, 1, 9, -9, 7, 2, -1, 4, 7, 0, 8, 13, 5, 3, 4, 6, 2, 6, 7, 5, -5, 5, 11, 5, 12, 6, 5, 6, 8, 8, 6, 12, 10, 12, 13, 0, 3, 3, 6, 2, 9, 8, 7, 4, 0, 7, -8, 6, 12, 5, 2, -6, 4, 12, 10, 8, 8, 0, -1, 5, 13, 15, 4, 11, 11, 8, 6, 2, 11, -21, 7, 12, 9, 3, 7, -3, 7, -6, 10, 7, -1, 10, 7, 5, 5, -5, 3, 10, 6, 4, 12, 12, 8, 4, 12, 10, 10, 0, 5, -2, -1, 12, 0, 9, -3, -8, 12, -2, 3, 6, 10, 10, -2, 5, 2, 7, -13, 5, 2, 6, 8, 9, 6, -4, 5, 10, 7, 6, 11, -7, 8, 8, -3, 8, 9, 11, -3, 5, 3, 7, 11, 9, 2, 6, 8, 3, 10, 10, 3, 8, 6, 8, 11, 7, -7, 7, 10, 7, -5, 12, 4, 9, 5, -4, 7, 7, 8, 8, 5, 9, 4, 11, 8, 5, 1, 5, 11, -5, 8, 6, 8, 9, 5, 7, 12, 5, -1, 6, 11, 8, 5, -7, 4, 10, 3, 3, 5, 7, 7, 6, 6, -11, 5, 4, 7, 3, 5, 8, 9, 10, 7, 7, 8, 8, 7, -9, 3, 8, -1, 7, -4, 11, 7, 11, 0, 6, 4, 0, 6, -13, 6, 12, 7, -9, 9, 6, 2, -4, 8, -12, 7, 10, 9, 6, -3, 3, 13, 13, 8, -1, 3, -5, 7, 7, 2, 7, 11, 10, 6, 1, -6, 5, -19, 5, 7, 4, 2, 7, 7, 6, 12, 5, 9, 5, 0, 0, 15, 12, -3, 6, 4, 7, 6, 4, 6, 10, 4, 2, 3, 2, 8, 0, 7, 5, 11, -5, 11, 5, 6, 6, 2, 9, 9, 10, -8, 2, -6, 5, 3, 12, -8, 7, 10, 5, 0, 4, -3, 4, 5, 11, 8, 3, 13, 8, 12, -1, 5, 6, 6, 5, 4, 8, 7, 9, 10, 8, -8, 8, 9, -7, 7, 4, 9, 8, 3, 10, 5, 5, 8, 13, 9, 10, 13, 6, 7, 6, 7, 7, 3, 9, 12, 6, 9, 3, 6, -9, 8, 8, -27, -4, 0, 4, 3, 8, -2, 4, 9, 9, 12, 9, -2, 2, 9, 10, 6, 6, 9, 8, 5, 7, 13, 13, 7, 7, 13, 8, -6, 7, 3, 2, 4, 5, 7, 6, 2, 5, -6, 3, 6, -6, 7, -1, -4, 8, 10, 8, 9, 8, 5, 2, 9, 7, 11, 13, 8, -7, 7, 4, 6, 10, 7, 6, 5, 6, 12, 4, 6, 9, 10, 3, 12, 8, 8, 2, 8, 6, 5, 9, 5, 6, -6, 7, 8, 5, 5, 7, 8, 10, -7, 9, 4, 4, 2, 5, 6, -2, 3, 8, 12, 11, 9, 13, 10, -6, 3, 5, 8, 6, 5, 12, -8, 8, 7, 7, 6, 2, 5, 8, 12, 11, 8, 5, 0, -2, -6, 6, 11, 8, 1, 12, 7, 11, 5, 4, 12, 6, 10, 6, 12, 6, 1, 10, 8, 8, -14, 5, 5, 6, 7, 7, 7, 4, 10, 7, 5, -7, -6, -6, 3, 6, -5, 10, -4, 5, 1, 3, 10, 4, 10, 13, 15, 11, 10, 4, 6, -1, 3, 7, 0, 5, 10, 6, 7, 5, 8, 6, 11, 5, 6, 7, 13, 6, 7, 6, 5, 3, 11, 1, 9, -5, 6, 6, 6, -4, 7, 6, 9, 6, -4, 11, 6, 3, 9, -6, 5, 3, 9, 2, 7, 6, 12, 9, 11, 6, 1, 7, 5, 8, 8, 3, 12, 5, 0, -2, 5, 7, 10, 8, -6, 4, 4, 6, 11, 4, -1, 6, 10, 6, 6, 8, 2, 4, 9, -2, 4, 12, 7, 5, 4, 7, 5, 4, 7, 7, 8, 4, 5, 5, 8, 8, 3, 6, 10, 9, 10, 6, 9, -20, 6, 7, 8, 8, 4, 1, -6, 11, 5, 7, 5, 4, 2, 9, 8, 6, 8, 8, -6, 8, 10, 7, 4, 8, 9, 7, 6, 7, 4, 7, 13, 6, -4, 6, 3, 5, 6, 9, 5, 11, 7, -4, 7, -11, 6, 4, 5, 9, 7, 3, 11, 6, 3, 8, 4, 5, 12, 11, 7, 7, 6, 5, 11, 2, 7, 5, 7, 6, 5, 9, 7, 9, 5, 0, 10, 12, 4, 2, 8, 5, 8, 8, 9, 10, 5, 6, -2, 10, 7, 3, -2, 4, 8, 5, -6, 9, 6, 10, 10, 9, 6, 10, 11, 2, 6, 10, 4, 8, 4, 4, -1, 7, 7, 5, 9, 12, 10, 4, 10, 8, 8, 5, 5, -3, 6, 6, 1, 7, 6, 7, 7, -5, 0, 5, -7, 14, 2, 2, 3, 11, -4, 5, 8, 13, 6, -2, 5, 8, 12, 5, 2, 10, 5, 2, 3, 10]


x = np.arange(len(y))

plt.plot(x, y)
plt.show()
plt.close()
