import getopt
import numpy as np
import os
import sys
import multiprocess
from joblib import Parallel, delayed
import imageio
import time

def crop_center(img,cropx,cropy):
    """
    Central cropping of array
    
    Params:
        img : np.ndarray
            Image to be centrally cropped
        cropx : int
            x size of cropping
            
    Returns: cropped image (np.ndarray)
    
    """
    
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def imageCropping(img,s):
    """
    Central image cropping
    
    Params:
        img : np.ndarray
            Image to be centrally cropped
        size : int or list of int
            Size of images to be obtained
            
    Returns: cropped image (np.ndarray)
    
    """
    if type(s) == list:
        if len(s) != 2:
            print("Given size has more than 2 dimensions")
            sys.exit(2)
        
        img_crop = crop_center(img, s[0], s[1])
        
    elif s > 0:
        img_crop = crop_center(img, s, s)
        
    else:
        img_crop = np.copy(img)
        
    return img_crop

def parallelAccess(path, size, out, proj = "max"):
    """
    Access folders to preprocess in parallel
    
    Params:
        - path : str
            Folder to be accessed
        - size : int or list of int
            Size of images to be obtained
        - out : str
            Parent folder where to save results
        - proj : str
            Projection type ("max" for maximum, "mean" for mean, "min" for minimum, "median" for median)
            
    Outputs:
        Saved MIPs and copied fluorescence images into "out" parent folder
    
    """
    
    if not(os.path.exists(path)):
        print("Subfolder {} does not exist".format(path))
        sys.exit(2)
       
    if os.path.isdir(path):
        files = np.array(sorted(os.listdir(path))) # Image .tif files
        if len(files) > 0:
            # Identify individual images to be processed
            ind_images = np.flatnonzero(np.core.defchararray.find(files,".tif")!=-1) # Index of .tif files
            ind_individuals = np.flatnonzero(np.core.defchararray.find(files,"A01Z01C01")!=-1) # File positions where a new "image" starts

            inds = np.intersect1d(ind_images, ind_individuals)

            for i in inds:
                image_name = files[i].replace("A01Z01C01.tif","") # Common title for all .tifs from an image
                output_fluor_folder = os.path.join(out, os.path.basename(path), image_name, "Fluor")
                output_mip_folder = os.path.join(out, os.path.basename(path), image_name, "MIP")
                if not(os.path.exists(output_fluor_folder)):
                    os.makedirs(output_fluor_folder)
                if not(os.path.exists(output_mip_folder)):
                    os.makedirs(output_mip_folder)
                image_files_ind = np.flatnonzero(np.core.defchararray.find(files,image_name)!=-1) # Common images positions in folder
                image_files = files[image_files_ind]
                stack_list = []
                cont_stack = 0
                for image_file in image_files:
                    if "C04" in image_file: # Brightfield, add to stack list
                        try:
                            stack_list.append(imageio.imread(os.path.join(path,image_file)))
                        except:
                            print("Image file '{}' could not be read".format(os.path.join(path, image_file)))
                            sys.exit(2)
                        if cont_stack == 0:
                            stack_filename = os.path.join(output_mip_folder, image_file)
                        cont_stack += 1
                    else: # Fluorescence, copy to output folder
                        fluor = imageio.imread(os.path.join(path,image_file))

                        # Cropping fluorescence
                        fluor_crop = imageCropping(fluor,size)

                        imageio.imwrite(os.path.join(output_fluor_folder,image_file),fluor_crop)


                # Obtain projection
                stack_array = np.array(stack_list)
                if proj.lower() == "max" or proj.lower() == "maximum":
                    mip = np.max(stack_array, 0)
                if proj.lower() == "min" or proj.lower() == "minimum":
                    mip = np.min(stack_array, 0)
                elif proj.lower() == "mean" or proj.lower() == "average":
                    mip = np.mean(stack_array, 0)
                elif proj.lower() == "median":
                    mip = np.median(stack_array, 0)

                # Cropping projection
                mip_crop = imageCropping(mip,size)

                imageio.imwrite(stack_filename, mip_crop)

            

def dataAccess(path, size, output_folder, key = "max"):
    """
    Provide Maximum Intensity Projection (MIP) of given images in folders, plus
    provide the corresponding flourescent target images
    
    Params:
        path : str
            Root folder from where to start examining
        size : int or list of int
            Size of images to be obtained
        output_folder : str
            Folder where to save MIP file as .npy (if "save" flag is True)
        key : str
            Type of projection to provide ("max" for maximum intensity projection, "mean" for mean intensity projection,
            "median" for median intensity projection)
            Default: "max"
            
    Outputs:
        mips : list of np.array
            List with MIP results [MIP_20x, MIP_40x, MIP_60x]
        targets : list of np.array
            List with fluorescent target images [ch01_images, ch02_images, ch03_images] --> ch01_images = [ch01_20x, ch01_40x, ch01_60x]
            
    """
    init = time.time()

    if not(os.path.exists(path)):
        print("Folder {} does not exist".format(path))
        
    folders = sorted(os.listdir(path)) # Read all folders in the given path
    Parallel(multiprocess.cpu_count())(delayed(parallelAccess)(os.path.join(path,folder), size, output_folder, key) for folder in folders)    
    
    print("Preprocessing completed! Time ellapsed: {} seconds".format(round(time.time() - init,2)))

    

def main(argv):
    
    """
    Preprocessing: read the raw dataset, and save projections of brightfield images 
    and fluorescent images in a structured folder like:
        - Parent folder:
            - Magnification type 1
                - Image 1
                    - Projection folder
                        - Brightfield projection files
                    - Fluorescence images folder
                        - Fluorescent images
                        
            And so on
            
    Params:
        - path : str
            Name of input raw folder
        - s : numeric str
            Size of images to be obtained
        - t : str
            Type of projection to be applied ("max": for maximum, "mean" for mean, "median": for median, "min": for min)
        - out : str
            Name of parent output folder with preprocessed data
    """
    
    path = ""
    out = ""
    s = ""
    t = ""
    
    try:
        opts, args = getopt.getopt(argv, "hp:s:t:o:r", ["path=","size=","type=","out="])
    except getopt.GetoptError:
        print("preprocessing.py -p <input_path> -s <size> -t <projection_type> -o <output_folder>")
        sys.exit(2)
       
    for opt, arg in opts:
        if opt == '-h':
            print("preprocessing.py -p <input_path> -s <size> -t <projection_type> -o <output_folder>")
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-s", "--size"):
            s = arg
        elif opt in ("-t", "--type"):
            t = arg
        elif opt in ("-o", "--out"):
            out = arg
    
    if path == "":
        print("preprocessing.py -p <input_path> -s <size> -t <projection_type> -o <output_folder>")
        sys.exit()
    
    if t == "":
        print("Unspecified projection type. Working by default with: 'maximum'")
        t = "max"
        
    if s == "": 
        print("Unspecified size. Working with the whole image...")
        s = 0
    else:
        try:
            if "," in s:
                s = s.split(",")
                s = [int(size) for size in s]
            else:
                s = int(s)
        except:
            print("Non-numeric input for size")
            sys.exit(2)
    
    if out == "":
        print("Unspecified output folder. Outputting results to current working directory...")
        out = os.getcwd()
        
    dataAccess(path, s, out, t)

if __name__ == '__main__':
    main(sys.argv[1:])    
