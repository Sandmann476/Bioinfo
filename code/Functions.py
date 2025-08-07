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



def nuclei_count(file_select, size_min, size_max):
    data = {}
    binary_list = []
    
    for file in file_select:
        data[file] = []
        image = ski.io.imread(file)
        # Apply Gaussian filter to denoise the image
        image_denoised = gaussian(image, sigma=2, preserve_range=True)
        image_binary = image_denoised > ski.filters.threshold_li(image_denoised)
        split_objects = split_touching_objects(image_binary)
        # using connected component labeling
        labeled_objects = ski.measure.label(split_objects, connectivity=1)  
        # exclude labels out of size range
        filtered_labels = cle.exclude_labels_outside_size_range(labeled_objects, minimum_size=size_min, maximum_size=size_max)  
        # for counting nuclei
        num_features = filtered_labels.max()  
        data[file].append(num_features)
        # Convert filtered labels to monochrome (binary) image
        monochrome = (filtered_labels > 0).astype(np.uint8)
        binary_list.append(monochrome)

#saving images as PDF
    
# Set your output directory and file name
    output_dir = r"C:\Users\grufl\Desktop\211007 Sicherung USB\uni\Prakt bioinfo\Bioinfo\data\processed/images"
    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(binary_list):
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
        ax.imshow(binary_list[i])
    plt.show()
    print(data)
    
