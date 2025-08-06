import skimage as ski
import numpy as np
import pandas as pd
from skimage.filters import gaussian
import glob
import matplotlib.pyplot as plt
from napari_segment_blobs_and_things_with_membranes import split_touching_objects
import os
import pyclesperanto as cle
from PIL import Image
from scipy import ndimage


file_select = glob.glob(r"C:\Users\grufl\Desktop\211007 Sicherung USB\uni\Prakt bioinfo\Bioinfo\data\raw\selected-tiles\out_opt_flow_registered_X10_Y10_*.tif")

def count_nuclei(file_select, sigma_select=2, size_min = 150, size_max = 2000):
    file_list = []
    for file in file_select:
        file_list.append(ski.io.imread(file))

    
    data = {
        "file": [],
        "nuclei_count": [],
        "binary_list": [],
    }

    # Set your output directory and file name
    output_dir = r"C:\Users\grufl\Desktop\211007 Sicherung USB\uni\Prakt bioinfo\Bioinfo\data\processed/images"
    os.makedirs(output_dir, exist_ok=True)

    for file in file_select:
        data["file"].append(file)
        image = ski.io.imread(file)
        # Apply Gaussian filter to denoise the image
        image_denoised = gaussian(image, sigma_select, preserve_range=True)
        image_binary = image_denoised > ski.filters.threshold_li(image_denoised)
        split_objects = split_touching_objects(image_binary)
        # Convert boolean array to uint8 for cle.gauss_otsu_labeling
        split_objects_uint8 = split_objects.astype(np.uint8)
        labeled_objects = cle.gauss_otsu_labeling(split_objects_uint8, outline_sigma=0)
        filtered_labels = cle.exclude_large_labels(cle.exclude_small_labels(labeled_objects, maximum_size=size_min), minimum_size=size_max)
        labeled_array, num_features = ndimage.label(filtered_labels)
        data["nuclei_count"].append(num_features)
        # Convert filtered labels to monochrome (binary) image
        monochrome = (filtered_labels > 0).astype(np.uint8)
        #nuclei_count.append(num_features)
        data["binary_list"].append(monochrome)

    #saving images as PDF

    for idx, img in enumerate(data["binary_list"]):
        # Erzeuge einen Dateinamen, z.B. monochrome_0.pdf, monochrome_1.pdf, ...
        pdf_path = os.path.join(output_dir, f"monochrome_{idx+1}.pdf")
        # Falls img ein pyclesperanto-Array ist, zuerst zu numpy holen:
        if hasattr(img, "get"):
            img = img.get()
        pil_img = Image.fromarray((img * 255).astype('uint8')).convert('L')
        pil_img.save(pdf_path)
        print(f"Gespeichert: {pdf_path}")

    # making a csv file
    df = pd.DataFrame(data)
    csv_file_path = r"C:\Users\grufl\Desktop\211007 Sicherung USB\uni\Prakt bioinfo\Bioinfo\data\processed\csv\analysis_results.csv"
    df.to_csv(csv_file_path, index=False)
    print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')

    fig, axs = plt.subplots(4, 6, figsize=(25, 15))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(data["binary_list"][i])
    plt.show()
    #print(nuclei_count)
    print(data)
    
    