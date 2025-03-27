##########################################################################IMPORT MODULES###############################################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import zipfile
import utilities as ut
from scipy.signal import convolve2d
import os
##########################################################################DODGING AND BURNING###############################################################################################################
"""
        Dodging and Burning
        ===================

        1. Adjust the exposure of specific areas of the image. (Applied to region with large contrast)
        2. Calculate Center Around function
            1. Create Gaussian Blur Matrix (3 * 3) with specific s and alphai(scaling factor) from eq 5.
                1. aplhai for v1 - 0.354
                2. alphai for v2 - 1.6 * 0.354
                3. A common rule is to use a kernel size that is atleast 6 times the standard deviation.This ensures that the weights at the edges of the kernel are close to zero.

            2. Convolution with the Luminance values and that is v1
            3. Repeat Step 1 and 2 with different scale (1.6s) and calculate v2
            4. subtracting v1 - v2 / (Normalizing factor) gives us the center around function.
            5. Normalizing Factor - a * ((2 ^ phi) / s**2)
                1. phi - 8
                2. a - key value (Per image basis) (Try with some default value)
            6. Check for optimzied scale which gives us the Center around function less than the threshold (0.05)
        3. Calculate the Gaussian Kernel with sm and do convolution with the image get V1 
        4. Calcuate display luminance 
            1. Ld(x, y ) = L(x,y) / (1 + V1(x,y , sm))
        5. Apply Ld on the image.
            1. Convert the Image to HSV format.
            2. Replace the V channel with update one.
            3. Color Convert it to RGB.
"""
class DodgingAndBurning:
    def __init__(self, imagePath):
        self.utObj = ut.Utilities(imagePath)

        self.scale_selection_value = 0.05
        self.scale_multiplying_factor = 1.6
        self.phi = 8
        self.key_value = 0.36
        self.alpha_1 = 0.354;
        self.alpha_2 = self.scale_multiplying_factor * self.alpha_1
        self.iteration = 6
        self.threshold_value_scale_calculation = 0.05
        self.gamma = 2
        self.delta = 1e-6
        self.key_val = 0.35
        self.scale = 2
        self.images = self.utObj.get_images() # small set of images storing in a list
        self.images = self.utObj.color_convert(self.images)
        self.images = self.utObj.gamma_correction_images(self.images, self.gamma) 
        self.images = self.utObj.normalize_images(self.images) # normalized rgb images   
    
    def calculate_log_average_luminance(self, images):
        log_average_luminance = []
        for luminance_values in images:
            log_avg_luminance = np.exp(np.mean(np.log(self.delta + luminance_values)))
            log_average_luminance.append(log_avg_luminance)
        return log_average_luminance
    
    def calculate_scaled_luminance(self , luminance_images , log_average_luminance):
        scaled_luminance_images = []
    
        for index in range(len(self.images)):
            scaled_luminance_val = (self.key_val / log_average_luminance[index]) * (luminance_images[index])
            scaled_luminance_images.append(scaled_luminance_val)
        return scaled_luminance_images
    
    def gaussian_kernel_convolution(self,luminance_image,scale,alpha, kernel_size):
        kernel = self.utObj.createGaussianKernel(kernel_size , alpha , scale)
        luminance_image= np.array(luminance_image)
        image_dimension = luminance_image.shape
        rows , _ = image_dimension[0] , image_dimension[1]
        #print(luminance_image)
        #padding = (rows - kernel_size + 1) // 2 
        padding = kernel_size // 2
        padded_image = np.pad(luminance_image , ((padding, padding) , (padding , padding)) , mode = 'constant')
        convoluted_image = convolve2d(padded_image , kernel , mode = 'same')
        # print(convoluted_image)
        # convoluted_image = np.zeros((rows , cols))
        # rows , cols = padded_image.shape[0] , padded_image.shape[1]
        # for i in range(rows):
        #     for j in range(cols):
        #         startIndexRow = i
        #         startIndexCol = j
        #         sum = 0
        #         for i_ in range(kernel_size):
        #             for j_ in range(kernel_size):
        #                 sum += padded_image[i + i_][j + j_] * gaussian_kernel[i_][j_]
        #         convoluted_image[i,j] = sum
        # print(convoluted_image)
        return convoluted_image
    def calculate_center_surround_function(self, luminance_image , scale):
        kernel_size = int(6 * scale)
        convolution_1 = self.gaussian_kernel_convolution(luminance_image ,scale , self.alpha_1, kernel_size)
        convolution_2 = self.gaussian_kernel_convolution(luminance_image , 1.6 * scale , self.alpha_2, kernel_size)
        center_surround_function = (convolution_1 - convolution_2) / (((self.key_value * (2 ** self.phi)) / (scale**2)) + convolution_1)
        return center_surround_function

    def find_optimum_scale(self,luminance_image):
        scale = self.scale
        optimum_scale = None
        for _ in range(self.iteration):
            center_surround_function = self.calculate_center_surround_function(luminance_image , scale)
            # print(center_surround_function)
            # print('*******************************************************************')
            # x = center_surround_function < self.threshold_value_scale_calculation
            # print(np.sum(x))
            # print(x.size - np.sum(x))
            print(_)
            if np.all(center_surround_function < self.threshold_value_scale_calculation):
                print("sdga")
                optimum_scale = scale
            scale = self.scale_multiplying_factor * scale
        return optimum_scale
    
    def calculation_optimum_scale(self, luminance_images):
        optimum_scales = []

        for index in range(len(luminance_images)):
            scale = self.find_optimum_scale(luminance_images[index])
            optimum_scales.append(scale)
        return optimum_scales

    def calculate_tone_map_operators(self, optimum_scales, luminance_images):
        tone_map_operators = []

        for index in range(len(optimum_scales)):
            center_surround_value = self.calculate_center_surround_function(luminance_images[index] , optimum_scales[index])
            center_surround_value_resized = cv2.resize(center_surround_value, (luminance_images[index].shape[1], luminance_images[index].shape[0]))
            tmo = luminance_images[index] / ( 1 + center_surround_value_resized)
            tone_map_operators.append(tmo)
        return tone_map_operators
    
    def generate_images_for_display(self, tone_map_operator_images , luminance_images):
        toned_mapped_images = []
        for index in range(len(self.images)):

            # image = self.utObj.rgb_images[index]
            # hsv_image = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)
            # hsv_image[:,:,2] = self.tone_map_operator_images[index] * (hsv_image[:,:,2] / self.utObj.luminance_images[index])
            # image_tone_mapped = cv2.cvtColor(hsv_image , cv2.COLOR_HSV2RGB)
          
            image_tone_mapped = np.zeros_like(self.images[index])
            image_tone_mapped[:,:,0] = tone_map_operator_images[index] * (self.images[index][:,:,0] / luminance_images[index])
            image_tone_mapped[:,:,1] = tone_map_operator_images[index] * (self.images[index][:,:,1] / luminance_images[index])
            image_tone_mapped[:,:,2] = tone_map_operator_images[index] * (self.images[index][:,:,2] / luminance_images[index])
            toned_mapped_images.append(image_tone_mapped)
        return toned_mapped_images

    
    def dodging_burning_calculation(self):
        luminance = self.utObj.luminance_calculation(self.images)
        luminance_log_average = self.calculate_log_average_luminance(luminance)
        luminance_scaled = self.calculate_scaled_luminance(luminance , luminance_log_average)
        optimum_scales = self.calculation_optimum_scale(luminance_scaled)
        
        print(optimum_scales)
        tone_map_operators = self.calculate_tone_map_operators(optimum_scales , luminance_scaled)
        tone_mapped_images = self.generate_images_for_display(tone_map_operators , luminance)   
        return tone_mapped_images

##########################################################################INITIAL LUMINANCE MAPPING###############################################################################################################
"""
        Initial Luminance Mapping
        ==========================

        1. Compute the log-average luminance of the image
            
            1. log average luminance - useful approximation to the key of the scene.
            2. Lw(x,y) - "World luminance" for pixel (x,y).
            3. N - Total number of pixels in the image.
            4. theta - small value to avoid singularity that occurs if black pixels are present in the image.

            Lw(bar) =1/N exp(Σ log( theta + Lw(x,y)))
        
        2. Compute the scaled luminance of the image
        
            1. L(x,y) - Scaled luminance for pixel (x,y).
            2. a - key value - Domain (0.045 -> 0.09 -> 0.18 -> 0.36 -> 0.72)
            3. "Key Value" - subjectively indicates if the image is light , dark , normal
            
            L(x,y) = a * Lw(x,y) / Lw(bar)
        
        3. Compute the tone mapping operator

            1. Ld(x,y) - Tone Mapping Operator for pixel (x,y).
            2. High luminance scaled by approx - 1 / L , while low luminance scaled by 1.

            Ld(x,y) = L(x,y) / (1 + L(x,y))
        4. Apply the tone mapping operator to the image
        
        5. Modified tone mapping operator

            Ld(x,y) = L(x,y) * (1 + (L(x,y) / (Lwhite)^2)) / (1 + L(x,y))

            1. Ld(x,y) - Modified Tone Mapping Operator for pixel (x,y).
            2. Lwhite -  smallest luminance that will be mapped to pure white.(set to the max luminance in the scene)
"""
class InitLuminanceMapping:
        ############################################### MEMBER VARIABLES ####################################################################################
    def __init__(self, imagePath):
        self.utObj = ut.Utilities(imagePath)
        self.luminance_images = []
        self.log_average_luminance = []
        self.scaled_luminance_images = []
        self.tone_map_operator_images = []
        self.modified_tone_map_operator_images = []
        self.toned_mapped_images = []
        self.modified_toned_mapped_images = []

        ############################################### CONSTANT VALUES ####################################################################################
        self.delta = 1e-6
        self.key_val = 0.78
        self.channels = 3
        self.gamma = 0.65
        ############################################## HELPER FUNCTIONS #####################################################################
        self.images = self.utObj.get_images() # small set of images storing in a list
        self.images = self.utObj.color_convert(self.images)
        #self.images = self.utObj.gamma_correction_images(self.images, self.gamma) 
        self.images = self.utObj.normalize_images(self.images) # normalized rgb images   
       
    
    def calculate_log_average_luminance(self, images):
        log_average_luminance = []
        for luminance_values in images:
            log_avg_luminance = np.exp(np.mean(np.log(self.delta + luminance_values)))
            log_average_luminance.append(log_avg_luminance)
        return log_average_luminance
    
    def calculate_scaled_luminance(self , luminance_images , log_average_luminance):
        scaled_luminance_images = []
    
        for index in range(len(self.images)):
            scaled_luminance_val = (self.key_val / log_average_luminance[index]) * (luminance_images[index])
            scaled_luminance_images.append(scaled_luminance_val)
        return scaled_luminance_images
    
    def calculate_modified_tone_map_operator(self , scaled_luminance_images , max_luminance_images):
        modified_tone_map_operator_images = []
        for index in range(len(scaled_luminance_images)):
            tone_map_operator_image = (scaled_luminance_images[index] * ( 1 + (scaled_luminance_images[index] / (max_luminance_images[index]**2))))/(1 + scaled_luminance_images[index])
            modified_tone_map_operator_images.append(tone_map_operator_image)
        return modified_tone_map_operator_images
    
    def calculate_tone_map_operator(self , scaled_luminance_images):
        tone_map_operator_images = []
        for index in range(len(scaled_luminance_images)):
            tone_map_operator_image = scaled_luminance_images[index] / (1 + scaled_luminance_images[index])
            tone_map_operator_images.append(tone_map_operator_image)
        return tone_map_operator_images

    def generate_images_for_display(self, tone_map_operator_images , luminance_images):
        toned_mapped_images = []
        for index in range(len(self.images)):

            # image = self.utObj.rgb_images[index]
            # hsv_image = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)
            # hsv_image[:,:,2] = self.tone_map_operator_images[index] * (hsv_image[:,:,2] / self.utObj.luminance_images[index])
            # image_tone_mapped = cv2.cvtColor(hsv_image , cv2.COLOR_HSV2RGB)
          
            image_tone_mapped = np.zeros_like(self.images[index])
            image_tone_mapped[:,:,0] = tone_map_operator_images[index] * (self.images[index][:,:,0] / luminance_images[index])
            image_tone_mapped[:,:,1] = tone_map_operator_images[index] * (self.images[index][:,:,1] / luminance_images[index])
            image_tone_mapped[:,:,2] = tone_map_operator_images[index] * (self.images[index][:,:,2] / luminance_images[index])
            toned_mapped_images.append(image_tone_mapped)
        return toned_mapped_images
    
    def generate_images_for_display_modified(self , modified_tone_map_operator_images , luminance_images):
        modified_toned_mapped_images = []
        for index in range(len(self.images)):
            image_tone_mapped = np.zeros_like(self.images[index])
            image_tone_mapped[:,:,0] = modified_tone_map_operator_images[index] * (self.images[index][:,:,0] / luminance_images[index])
            image_tone_mapped[:,:,1] = modified_tone_map_operator_images[index] * (self.images[index][:,:,1] / luminance_images[index])
            image_tone_mapped[:,:,2] = modified_tone_map_operator_images[index] * (self.images[index][:,:,2] / luminance_images[index])
            modified_toned_mapped_images.append(image_tone_mapped)
        return modified_toned_mapped_images
    
    def init_luminance_mapping_process(self):
        luminance = self.utObj.luminance_calculation(self.images)
        luminance_log_average = self.calculate_log_average_luminance(luminance)
        luminance_scaled = self.calculate_scaled_luminance(luminance , luminance_log_average)
        tone_map_operator = self.calculate_tone_map_operator(luminance_scaled)
        tone_mapped_images = self.generate_images_for_display(tone_map_operator , luminance)   
        return tone_mapped_images
    
    def init_luminance_mapping_process_modified(self):
        luminance = self.utObj.luminance_calculation(self.images)
        luminance_log_average = self.calculate_log_average_luminance(luminance)
        luminance_scaled = self.calculate_scaled_luminance(luminance , luminance_log_average)
        luminance_max = self.utObj.calculate_max_luminance(luminance_scaled)
        tone_map_operator_modified = self.calculate_modified_tone_map_operator(luminance_scaled, luminance_max)
        tone_mapped_images = self.generate_images_for_display_modified(tone_map_operator_modified ,luminance)
        return tone_mapped_images
         
#########################################################################################################################################################################################  
def load_images_from_folders(folders):
    folder_images = []
    for folder in folders:
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        folder_images.append(images)
    return folder_images

def show_images_in_grid(folder_images):
    num_folders = len(folder_images)
    max_images = max(len(images) for images in folder_images)
    
    fig, axes = plt.subplots(num_folders, max_images, figsize=(15, 15))
    
    for row, images in enumerate(folder_images):
        for col in range(max_images):
            ax = axes[row, col] if num_folders > 1 else axes[col]
            if col < len(images):
                ax.imshow(cv2.cvtColor(images[col], cv2.COLOR_BGR2RGB))
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

#########################################################################################################################################################################################          
def main():
    imagePath = "./resources/input/toneDataSet.zip"
    outputPath = "./resources/output"
    algo1 = "init_luminance_mapping"
    algo2 = "init_luminance_mapping_modified"
    ILM1 = InitLuminanceMapping(imagePath)
    images = ILM1.init_luminance_mapping_process()
    ILM1.utObj.write_images(images, outputPath, algo1)
    #print(ILM1.utObj.is_image_all_black(images[0]))
    # image = (images[0] * 255).astype(np.uint8)
    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Image BGR', image_bgr)
    # #cv2.imshow('Image', images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ILM2 = InitLuminanceMapping(imagePath)
    images = ILM2.init_luminance_mapping_process_modified()
    ILM2.utObj.write_images(images, outputPath, algo2)

    algo1 = "dodge_burning"
    DB = DodgingAndBurning(imagePath)
    images = DB.dodging_burning_calculation()
    DB.utObj.write_images(images, outputPath, algo1)

    
    folders = ["./resources/output/dodge_burning", "./resources/output/init_luminance_mapping", "./resources/output/init_luminance_mapping_modified"]

    # Load images from the folders
    folder_images = load_images_from_folders(folders)

    # Show images in a grid
    show_images_in_grid(folder_images)  

    

def main2():
    imagePath = "./resources/input/toneDataSet.zip"
    image  = [[1,2,3],[3,4,5],[6,7,8]]
    kernel = [[1,1,1],[1,1,1],[1,1,1]]

    img1 = DodgingAndBurning(imagePath)
    img2 = img1.gaussian_kernel_convolution(image , kernel , 1 , 1)
    # print(img2)
if __name__ == "__main__":
    main()
    #main2()



