from PIL import Image
import os
import shutil

# Paths
dataset1 = 'C:/Users/wimuk/Desktop/Dataset_BUSI_with_GT'
dataset2 = 'C:/Users/wimuk/Desktop/BUS/organized'
target_folder = 'C:/Users/wimuk/Desktop/Updated_combine_datasets'

def copy_and_rename_images(dataset2, target_folder, label, start_index=1):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(dataset2):
        if filename.endswith('.png') and '_mask' not in filename:
            new_filename = f"{start_index}.png"
            shutil.copy(os.path.join(dataset2, filename), os.path.join(target_folder, new_filename))
            start_index += 1

# Example usage for Dataset 1
# copy_and_rename_images(dataset1 + '/benign', target_folder + '/benign', 'benign', start_index=1)
# copy_and_rename_images(dataset1 + '/malignant', target_folder + '/malignant', 'malignant', start_index=1)
# copy_and_rename_images(dataset1 + '/normal', target_folder + '/normal', 'normal', start_index=1)

# # Continue with Dataset 2
# copy_and_rename_images(dataset2 + '/Benign', target_folder + '/benign', 'benign', start_index=438) # 437 benign images from Dataset 1
copy_and_rename_images(dataset2 + '/Malignant', target_folder + '/malignant', 'malignant', start_index=211) # 210 malignant images from Dataset 1