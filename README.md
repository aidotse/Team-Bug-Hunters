# Team-Bug-Hunters

The provided code for the hackathon is composed by the following files. To train the proposed neural network from scratch for three different magnification levels and fluores-cent channels, the preprocessing.pypython file needs to be executed which arrangesthe dataset in the desired format and then only the main.pyscript should be run which calls all the required definitions.

- main.py: To train the network from scratch, only this script needs to be exe-cuted which calls all the definitions and gives the prediction on the merged tiles.The trained model is saved in hdf5 format for each channel and magnification level and the checkpoints are also saved in the script.  If the user wants to train the network from saved checkpoints then the saved models should be located in a direc-tory named astraining_checkpoints, and the checkpoint should have a name of theformat:checkpoint_attUNet{magnification number}_fluor{number of fluorescencechannel}

- test_TeamBugHunters.py (FOR RUNNING THE TEST DATASET): this function performs the inference of our processfrom a given input folder. The library getopt was used to allow the user to inputa folder where images for testing are located. Optionally, the user can provide anoutput folder name for saving the images there. 

The input to be provided in the terminal for test_TeamBugHunters.py is:

    python3 test_TeamBugHunters.py -i input_folder -o output_folder

IMPORTANT:
1) The code should be in the same folder as utils.py,params.py, and merge_and_tile.py. 
2) All the models saved as .hdf5 (in the Drive folder) should also be in the same folder as this script and the other scripts (utils.py, params.py, and merge_and_tile.py). 

The code examines all the .tif files under the input folder, checks the magnification each file belongs to, and sorts all the .tif filepaths for magnification (getting only brightfield images with the tags A04 and C04). The image stack files are recognized by the beginning of the filename up to A04. When all files from the same magnification and stack are identified, the MIPs are obtained, being down-sampledto a size of 2048x2048. Then, the MIPs are tiled with 512x512 size and inputted tothe models. All images from the same magnification are processed at once, havingto load only the models that work with that magnification. The predictions are then generated and saved in a different subfolder for each magnification as uint16 datatype.

From the function description in the script (test_TeamBugHunters.py) (more detailed):

    File testing: read the test dataset with brightfield images to output the correspondent fluorescent predictions. The 
        code is fit to run the following folder structure:
            - Parent folder:
                - Magnification type 1 (20x_images)
                    - Set of brightfield images, separated into different Z slices
                - Magnification type 2 (40x_images)
                    ...

                And so on

        The model files used in the prediction SHOULD BE LOCATED IN THE SAME FOLDER AS THIS CODE

        The corresponding predictions will be saved in a folder that can be inputted by the user. If no output directory is 
        inputted by the user, the folder with the predictions will be created in the same directory as this code with the
        name "predictions_TeamBugHunters". The structure of the output folder will be the same as the structure of the input folder, 
        but with the prediction files (output filenames will be "AssayPlate_Greiner_#655090_xxx_TxxxFxxxLxxA0cZ01C0c.tif",
        with "_xxx_TxxxFxxxLxx" given by each set of Z stack and "c" given by the fluorescence channel that is analyzed)


        Params:
            - path : str
                Name of input test folder
            - out : str (optional)
                Name of output folder with results

        Outputs:
            void
            Saved prediction files as .tiff format in the output folder


- preprocessing.py: The file consist of cropping definition which is designed to cropthe input images and targets in a square shape of 2048x2048. Additionally, codeto access the dataset is written which takes the brightfield images and performsthe MIPs. Further the MIPs and the corresponding fluorescent images are saved instructured folders.

- data_loading.py: Once the dataset is saved in desired format the next step is toload the preprocessed data, create tiles of size 512x512 for each input image and itscorresponding target.

- merge_and_tile.py: The script takes numpy array as input and creates tiles. Toevaluate the models performance on the whole resolution image, merging code isalso implemented which merges the arrays of the tiles back into whole resolutionimage.

- utils.pyThe file consists of metrics, SSIM and PSNR which were used to evaluatethe model performance.

- model.py: The Residual U-Net model with Attention Connections is implementedin this python file which takes the tiles from MIPs in form of array along withcorresponding targets.

- params.py: The parameter list is prepared to carry out the preprocessing andtraining tasks. Along with the different parameter settings, the file also consistsof variablesfluor_chandmagnificationwhich are set as 1 and 20 respectively bydefault. To train the network for other channels and magnification, only these twovariables need to be altered without making any changes in themain.pyscript. Wherefluor_chset as 1 corresponds to Nuclei; 2 and 3 for Lipids and Cytoplasm,respectively. Similarlymagnificationset as 20 corresponds to 20x and 40, 60 corre-sponds to 40x and 60x, respectively. 



 
