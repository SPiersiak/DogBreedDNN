from tkinter import *
import Img as img
import test as test
import gc

root = Tk()

root.title('Dog Breed')
root.geometry("1920x1080")

# Define background image
bg = PhotoImage(file="images/bg.png")
# Create label
main_label = Label(root, image=bg)
second_label = Label(root, image=bg)


def main_window():
    """
    Display main window
    """

    def move_to_second_label():
        """
        Move User to second view
        """
        # Take image form user class
        z, path = img.upload_image()
        if z is not None:
            value, possible = test.recognize_image(path)
            if value is not None:
                main_label.pack_forget()
                second_label.pack()
                second_window(z, value, possible)

    main_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Create text
    banner_text = Label(main_label, text="You don't know what breed you dog is? Check it out now!",
                        font=("SimSun", 50), fg="black", bg="#f4ead9")
    banner_text.place(relx=0.5, rely=0.1, anchor="center")

    g_text = Label(main_label, text="",
                   font=("SimSun", 50), fg="black", bg="#faf0de")
    g_text.place(relx=0.5, rely=0.3, anchor="center")

    # Create button
    take_photo_button = Button(main_label, text="Insert a picture of your dog.", font=("Segoe UI", 50), bg='#fffff5',
                               activebackground='#fffff5', command=move_to_second_label)
    take_photo_button.place(relx=0.5, rely=0.75, anchor="center")

    main_label.pack()
    root.mainloop()


def second_window(im, name, possible_name):
    """
    Display second window
    """

    def move_to_first_label():
        """
        Move User to first view
        """
        a_text.config(text="")
        w_text.config(text="")
        second_label.pack_forget()
        main_label.pack()
        main_window()
        second_label.destroy()

    second_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Create text
    b_text = Label(second_label, text="Your dog is ...",
                   font=("SimSun", 50), fg="black", bg="#faf0de")
    b_text.place(relx=0.75, rely=0.1, anchor="center")

    # Place for image
    image_holder = Label(second_label, image=im, height=500, width=500)
    image_holder.image = im
    image_holder.place(relx=0.2, rely=0.5, anchor="center")

    # Text attribute - a place for the answer from the algorithm
    a_text = Label(second_label, text=name,
                   font=("SimSun", 35), fg="black", bg="#fdf3e1")
    a_text.place(relx=0.60, rely=0.5, anchor="w")

    # Text attribute - a place for the possible dog breed
    y_text = Label(second_label, text="It can also be:",
                   font=("SimSun", 40), fg="black", bg="#faf0de")
    y_text.place(relx=0.5, rely=0.8, anchor="center")
    w_text = Label(second_label, text=possible_name[1] + ", " + possible_name[2] + ", " + possible_name[3],
                   font=("SimSun", 30), fg="black", bg="#fdf3e1")
    w_text.place(relx=0.1, rely=0.9)
    w_text.config(anchor=CENTER)

    # Back Button
    back_button = Button(second_label, text="Back", font=("Segoe UI", 40), bg='#fffff5',
                         activebackground='#fffff5', command=move_to_first_label)
    back_button.place(relx=0.1, rely=0.07, anchor="center")
    second_label.pack()
    gc.collect()
