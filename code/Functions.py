import skimage as ski
from skimage.filters import gaussian
import glob
import matplotlib.pyplot as plt
from napari_segment_blobs_and_things_with_membranes import split_touching_objects


def count_nuclei(xstart, ystart):
    file_select = glob.glob(f"C:\Users\grufl\Desktop\211007 Sicherung USB\uni\Prakt bioinfo\Bioinfo\data\raw\selected-tiles\out_opt_flow_registered_X{xstart}_Y{ystart}_*.tif")
    file_list = []
    for file in file_select:
        file_list.append(ski.io.imread(file))

    fig, axs = plt.subplots(4, 6, figsize=(25, 15))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(file_list[i])
    plt.show()

    binary_list = []
    for image in file_list:
        image_denoised = gaussian(image, sigma=2, preserve_range=True)
        image_binary = image_denoised > 130
        split_objects = split_touching_objects(image_binary)
        binary_list.append(split_objects)

    fig, axs = plt.subplots(4, 6, figsize=(25, 15))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(binary_list[i])
    plt.show()
    