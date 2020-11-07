
# coding: utf-8

# In[ ]:


def tile(image):
    """
    This function creates tiles from the folder "MIP Images" 
    and saves the tiles in png format to folder "MIP_Images_Tiles" 
    """
    
    PATH = '/home/gauss/Documents/ACIC/MIP Images/'
    prefix = image.split(".tif")[0]
    image_path = os.path.join(PATH,image)   
    tiles = image_slicer.slice(image_path,16,save=False)
    image_slicer.save_tiles(tiles, directory='/home/gauss/Documents/ACIC/MIP_Images_Tiles/', prefix=prefix, format='tiff')


# In[ ]:


#tiling the MIP images by calling the 'tile function'
mip_images = sorted([os.path.basename(x) for x in glob.glob('/home/gauss/Documents/ACIC/MIP Images/*.tif')])
for i in range (len(mip_images)):
    mip_image = mip_images[i]
    tile(mip_image)
print(len(sorted([os.path.basename(x) for x in glob.glob('/home/gauss/Documents/ACIC/MIP_Images_Tiles/*.tiff')])))


