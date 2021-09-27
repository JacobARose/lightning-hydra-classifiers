"""

lightning_hydra_classifiers/utils/report_utils/pandas_embed_images.py

Author: Jacob A Rose
Created: Friday Sept 17th, 2021

TODO:

    [_] Add argparse section to allow calling as an executable script and pointing to a csv file to read & process from the cmdline.

"""

__all__ = ["df_embed_paths2imgs"]


import PIL
from pathlib import Path
import pandas as pd
from IPython.core.display import HTML
import base64
from io import BytesIO
from typing import Optional


def image_formatter(img):
    def image_base64(img):
        if isinstance(img, (Path, str)):
            img = PIL.Image.open(img)
        with BytesIO() as buffer:
            img.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
        
    return f'<img src="data:image/jpeg;base64,{image_base64(img)}">'


def df_embed_paths2imgs(df: pd.DataFrame,
                        file_path: str, 
                        path_col: str="path",
                        display: bool=False
                       ) -> Optional[HTML]:
    """
    Takes a dataframe with one column containing image file paths, embeds the raw bytes from each image into the dataframe and saves as a single  HTML file. Optionally outputs an IPython.core.display import HTML
    
    TODO:
        [_] Add option to embed much simpler html reference to image paths, rather than the raw image bytes. Allow referencing remotely hosted image repos without creating a massive single HTML file. (Added task Saturday Sept 18th, 2021)
    
    """
    
    if not file_path.endswith(".html"):
        file_path = f"{file_path}.html"
    df.to_html(file_path,
               escape=False,
               formatters={path_col:image_formatter}
              )
    if display:
        return HTML(filename=file_path)







# if __name__ == "__main__":
    
    
    