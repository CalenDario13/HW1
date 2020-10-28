# from collections import defaultdict
# import numpy as np
# from numpy import histogram as hist
# from PIL import Image
# from numpy import histogram as hist  # call hist, otherwise np.histogram
# import matplotlib.pyplot as plt
# import histogram_module
#
#
# def rgb2gray(rgb):
#
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray
#
#
# def normalized_hist(img_gray, num_bins):
#     assert len(img_gray.shape) == 2, 'image dimension mismatch'
#     assert img_gray.dtype == 'float', 'incorrect image type'
#     pixel_value_dict = defaultdict(int)
#     for row in img_gray:
#         for pixel in row:
#             pixel_value_dict[pixel] += 1
#     bins=np.linspace(start=0,stop=255,num=num_bins+1)
#     hists=np.zeros(num_bins,dtype=int)
#     ordered_values=sorted(pixel_value_dict.keys())
#     for idx,(range_1, range_2) in enumerate(zip(bins[0:-1], bins[1:])):
#         hist_value_holder = 0
#         for value in ordered_values:
#             if value >= range_1 and value < range_2:
#                 hist_value_holder += pixel_value_dict[value]
#             elif value >range_2:
#                 break
#         hists[idx] = hist_value_holder
#
#     return hists, bins
#
# img_color = np.array(Image.open('./model/obj100__0.png'))
# img_gray = rgb2gray(img_color.astype('double'))
# num_bins_gray=40
#
# plt.figure(1)
# plt.subplot(1,3,1)
# plt.imshow(img_color)
#
# plt.subplot(1,3,2)
# hist_gray2, bin_gray2 = histogram_module.normalized_hist(img_gray, num_bins_gray)
# plt.bar((bin_gray2[0:-1] + bin_gray2[1:])/2, hist_gray2)
#
# plt.subplot(1,3,3)
# hist_gray1, bin_gray1 = hist(img_gray.reshape(img_gray.size), num_bins_gray,(0,255))
# plt.bar((bin_gray1[0:-1] + bin_gray1[1:])/2, hist_gray1)
#
# plt.show()

import numpy as np
x=np.zeros((3,4))
x[0,1]=2
print(x)
def check(elemnt):
    if elemnt == 2:
        return 3
    else:
        return 0
# for idx1,x_in in enumerate(x):
#     for idx2,element in enumerate(x_in):
#         if element == 2 :
#             x[idx1,idx2] = 3

print(list(map(check,x.reshape(-1))))
