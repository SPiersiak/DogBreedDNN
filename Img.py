from tkinter import filedialog
from PIL import Image, ImageTk

DEFAULT_RESOLUTION = (500, 500)


def upload_image():
    """
    Upload image from local storage
    :return: Selected Image, File path
    """
    uploaded_image = None
    ret = uploaded_image
    try:
        file_path = filedialog.askopenfilename(filetypes=(('png files', '*.png'), ('jpg files', '*.jpg')))
        if file_path:
            uploaded_image = Image.open(file_path)
            uploaded_image = uploaded_image.resize(DEFAULT_RESOLUTION, Image.ANTIALIAS)
            ret = ImageTk.PhotoImage(uploaded_image)

    except:
        pass
    return ret, file_path
