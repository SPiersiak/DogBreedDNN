from tkinter import filedialog
from PIL import Image, ImageTk

DEFAULT_RESOLUTION = (330, 330)


def upload_image():
    uploaded_image = None
    try:
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        uploaded_image = uploaded_image.resize(DEFAULT_RESOLUTION, Image.ANTIALIAS)
        ret = ImageTk.PhotoImage(uploaded_image)

    except:
        pass
    return ret
