import os
import shutil
import cv2
import numpy as np
import zipfile

class Utilities:
    def __init__(self,imagePath):
        self.imagePath = imagePath
        self.delta = 1e-6
        self.channels = 3

    ########################################################################################################
    def get_images(self):
        images = []
        with zipfile.ZipFile(self.imagePath, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.png' , '.jpg' , '.jpeg' , '.gif' , '.bmp')):
                    with zip_ref.open(file_name) as file:
                            file_bytes = np.asarray(bytearray(file.read()) , dtype = np.uint8)
                            image = cv2.imdecode(file_bytes , cv2.IMREAD_COLOR)
                            if image is not None:
                                images.append(image)
        
        print(f"Total Images Loaded : {len(images)}")
        print(images[0].shape[:2])
        return images

    def write_images(self,images , path , algo):
        # Check if the directory exists
        output_dir = os.path.join(path, algo)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create a new directory
        os.makedirs(output_dir)

        # Write images to the new directory
        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f'image_{i}.jpg')
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(img_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            #img = (img * 255).astype(np.uint8)
            #cv2.imwrite(img_path, img)


    # def normalize_image(self , image):
    #     normalized_image = np.zeros_like(image)
    #     for ch in range(self.channels):
    #         image_channel = image[: , : , ch]
    #         image_max = image_channel.max()
    #         image_min = image_channel.min()
    #         print(image_max , image_min)
    #         normalized_image[: , :  , ch] = (image_channel - image_min) / (image_max - image_min)
    #     return normalized_image
    
    def normalize_image(self , image):
        normalized_image = np.zeros_like(image)
        normalized_image = image/ 255.0
        return normalized_image

    def normalize_images(self, images):
        normalized_images = []
        for image in images:
            normalized_image = self.normalize_image(image)
            normalized_images.append(normalized_image)
        return normalized_images
    
    def color_convert(self, images):
        rgb_images = []
        for image in images:
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            rgb_images.append(image)
        return rgb_images
    
    def luminance_calculation(self, images):
        """
            Obtaining luminance values = L = 0.27R + 0.67G + 0.06B
        """
        luminance_images= []
        for image in images:
            l_w = 0.27 * image[:,:,0] + 0.67 * image[:,: , 1] + 0.06 * image[:,:,2] + self.delta
            luminance_images.append(l_w)
        return luminance_images
    
    def calculate_max_luminance(self,scaled_luminance):
        max_luminance_images = []
        for index in range(len(scaled_luminance)):
            max_luminance = np.max(scaled_luminance[index])
            max_luminance_images.append(max_luminance)
        return max_luminance_images
    
    def createGaussianKernel(self, size, alpha=1, scale=1):
        gaussianKernel = np.zeros((size , size))
        center = size // 2
        # print(alpha ,  scale)
        for x in range(size):
            for y in range(size):
                diffx = x - center
                diffy = y - center

                gaussianKernel[x][y] = np.exp(-(diffx**2 + diffy**2) / (alpha * scale)**2)
        
        gaussianKernel/=(1/(np.pi * (alpha*scale)**2))
        gaussianKernel/=(np.sum(gaussianKernel))
        return gaussianKernel

    def gamma_correction(self , image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((index / 255.) ** inv_gamma) * 255 for index in np.arange(0 , 256)]).astype("uint8")
        #image_8bit = (image * 255).astype(np.uint8)
        return cv2.LUT(image, table)

    def gamma_correction_images(self,images, gamma):
        gamma_corrected_images = []
        for image in images:
            gamma_corrected_images.append(self.gamma_correction(image , gamma))
        return gamma_corrected_images

    