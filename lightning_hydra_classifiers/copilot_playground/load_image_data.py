Synthesizing 10/10 solutions

=======

def load_image_data(root_dir: str):
    """Loads the image data from the given root directory.

    Args:
        root_dir (str): The root directory to load the image data from.

    Returns:
        images (list): The list of image data.
    """
    images = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images

=======

def load_image_data(root_dir: str):
    """
    Loads all images from the given root directory into a list of numpy arrays.
    :param root_dir: The directory to load images from.
    :return: A list of numpy arrays containing the images.
    """
    data = []
    for filename in os.listdir(root_dir):
        img = cv2.imread(os.path.join(root_dir, filename))
        data.append(img)
    return data

=======

def load_image_data(root_dir: str):
    """
    Loads the image data from the root directory.

    :param root_dir: The root directory to search for image files.
    :return: A list of image files.
    """
    image_data = []
    for root, dirs, files in os.walk(root_dir):
        for file_ in files:
            path = os.path.join(root, file_)
            image_data.append(path)

    return image_data

=======

def load_image_data(root_dir: str):
    """
    Loads all of the images from the given directory into a list of numpy arrays.
    :param root_dir: The directory to load the images from
    :return: A list of numpy arrays of the images
    """
    image_list = []
    for filename in glob.glob(root_dir + '/*.jpg'):
        image_list.append(np.array(Image.open(filename)))
    return image_list

=======

def load_image_data(root_dir: str):
	"""
	Loads the image data from the root directory.
	
	:param root_dir: The path to the root directory.
	:return: A dictionary of the image data.
	"""
	
	# Import packages
	import os
	import numpy as np
	import pandas as pd
	from PIL import Image
	
	# Init variables
	data_dict = {}
	img_width = 0
	img_height = 0
	
	# Get list of files
	for root, dirs, files in os.walk(root_dir):
		
		# Loop through files
		for file in files:
			
			# Init image data
			img_data = []
			
			# Get image path
			img_path = os.path.join(root, file)
			
			# Open image
			img = Image.open(img_path)
			
			# Get image size
			img_width, img_height = img.size
			
			# Loop through image pixels
			for i in range(img_width):
				for j in range(img_height):
					
					# Get pixel data
					r, g, b = img.getpixel((i, j))
					
					# Append to image data
					img_data.append([r, g, b])
			
			# Convert to numpy array
			img_data = np.array(img_data)
			
			# Get image class
			img_class = file.split('.')[0]
			
			# Add to data dictionary
			data_dict[img_class] = img_data
	
	# Convert to pandas dataframe
	data_df = pd.DataFrame.from_dict(data_dict, orient='index')
	
	# Return data

=======

def load_image_data(root_dir: str):
    """Loads image data from root_dir into a list of tuples (image, label).
    
    Parameters
    ----------
    root_dir : str
        The root directory of the image data.
    
    Returns
    -------
    list
        A list of tuples of (image, label).
    """
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(subdir, file)
                label = os.path.basename(subdir)
                data.append((image_path, label))
    return data

=======

def load_image_data(root_dir: str):
    """
    Loads the data from the root directory.
    :param root_dir: The root directory to load the data from.
    :return: The data frame containing the data.
    """
    # Get the list of all the files and folders in the root directory.
    dir_list = os.listdir(root_dir)
    # Create a list of all the paths to the files in the root directory.
    path_list = [
        os.path.join(root_dir, x) for x in dir_list
    ]
    # Create a list of tuples. Each tuple contains the path to the image
    # and the image number.
    image_path_list = []
    for path in path_list:
        # Get the list of all the files in the directory.
        file_list = os.listdir(path)
        # Create a list of tuples. Each tuple contains the path to the image
        # and the image number.
        file_path_list = [
            (os.path.join(path, x), x.split('.')[0]) for x in file_list
        ]
        # Add the list of tuples to the main list.
        image_path_list += file_path_list
    # Create a data frame from the data.
    df = pd.DataFrame(image_path_list, columns=['image', 'image_number'])
    return df

=======

def load_image_data(root_dir: str):
	"""
	Loads the image data from the root directory.

	Args:
		root_dir (str): The directory to search for images.

	Returns:
		list: A list of the images and their labels.
	"""
	# The root directory should have two sub-directories: '0' and '1'.
	# The '0' directory should contain images with no lightning.
	# The '1' directory should contain images with lightning.
	# Each image should be a png image with a label in the filename.
	# For example: 'lightning_0_1.png' is an image of no lightning.
	# 'lightning_1_1.png' is an image of lightning.

	# TODO: Implement this function.

	return None

=======

def load_image_data(root_dir: str):
    """
    Loads all images in the root directory and returns a list of the image data

    Parameters
    ----------
    root_dir : str
        Path to the root directory of image data

    Returns
    -------
    list
        A list of image data in the form of numpy arrays
    """

    image_data = []
    for filename in glob.glob(f'{root_dir}/*.png'):
        image_data.append(
            imageio.imread(filename)
        )
    return image_data

=======

def load_image_data(root_dir: str):
    """
    Loads a .csv file containing image paths and their corresponding labels.

    Args:
        root_dir (str): The path to the root directory of the data.

    Returns:
        A tuple containing the image paths and their corresponding labels.
    """

    # Load the data
    data = pd.read_csv(os.path.join(root_dir, 'github_copilot_data_test.csv'),
                       header=None,
                       names=['path', 'label'])

    # Return the data
    return data['path'].tolist(), data['label'].tolist()
