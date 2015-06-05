

#generate Gaussian pyramid for A
def gaussian_pyramid(img, levels=3):
    G = img.copy()
    pyr = [G]
    for i in xrange(levels):
        G = cv2.pyrDown(G)
        print G.shape
        pyr.append(G)
        plt.subplot(levels, 1, i)
        plt.imshow(cv2.cvtColor(G, cv2.COLOR_BGR2RGB))
    plt.show()
    return pyr

A = cv2.imread(im_loc)
gaussian_pyramid(A, 3)