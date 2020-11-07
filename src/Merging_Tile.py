# %%

# coding: utf-8

# %%


#Converting the tiles into array
X_preds=[]
mip_images_tiles = sorted([os.path.basename(x) for x in glob.glob('/home/gauss/Documents/ACIC/MIP_Images_Tiles/*.tiff')])
len_mip_images_tiles = len(mip_images_tiles)
for j in range (len_mip_images_tiles):
    PATH = '/home/gauss/Documents/ACIC/MIP_Images_Tiles/'
    mip_image_tile = mip_images_tiles[j]
    tile = os.path.join(PATH,mip_image_tile)   
    tile_ = Image.open(tile)
    tile_array = np.asarray(tile_)
    X_preds.append(tile_array)
len(X_preds) 


# %%


def merge_tile(predicted_tiles):
    """
    This function merges the predicted tiles back to whole image of dimension
    2156*2256 and saves them in tiff format for evaluation" 
    """
    file = '/home/gauss/Documents/ACIC/Merged_Tiles/'
    n = len(X_preds)    #total number of arrays
    while n > n-32:     #n-?=16
        n = n-32
        print(n)
        arr1 = np.concatenate((X[n-16:n-12]), axis=1)
        arr2 = np.concatenate((X[n-12:n-8]), axis=1)
        arr3 = np.concatenate((X[n-8:n-4]), axis=1)
        arr4 = np.concatenate((X[n-4:n]), axis=1)
        a = np.vstack((arr1, arr2, arr3, arr4))
        im=Image.fromarray(a)
        im.save(file + 'merged_tile_{}.tiff'.format(n)) 
        if n ==48:      #when n is equal to total number of arrays then terminate the loop
            break
        n = n+48        #increasing the value of n to get other set of tiles  


# %%
def merge_tile_perimage(predicted_tiles):
    """
    This function merges the predicted tiles back to whole image of dimension
    2156*2256 and saves them in tiff format for evaluation" 
    """
    #file = '/home/gauss/Documents/ACIC/Merged_Tiles/'
    #n = len(X_preds)    #total number of arrays
    #while n > n-32:     #n-?=16
        #n = n-32
    arr1 = np.concatenate((X[0:4]), axis=1)
    arr2 = np.concatenate((X[4:8]), axis=1)
    arr3 = np.concatenate((X[8:12]), axis=1)
    arr4 = np.concatenate((X[12:]), axis=1)
    a = np.vstack((arr1, arr2, arr3, arr4))
    plt.imshow(a)
    plt.show()
    print(a)
    im=Image.fromarray(a)
    plt.imshow(im)
    #if n ==48:      #when n is equal to total number of arrays then terminate the loop
        #break
    #n = n+48        #increasing t


# %%
merge_tile(X_preds)   #calling the 'Tile Merging Function'


# %%
merge_tile_perimage(X_preds)   #calling the 'Tile Merging Function per image'
