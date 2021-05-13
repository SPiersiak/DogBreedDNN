from tkinter import *
import Img as img

root = Tk()
# Define min and max window size
root.minsize(1100, 650)
root.maxsize(1920, 1080)
root.title('Dog Breed')
root.geometry("1920x1080")

# Define background image
bg = PhotoImage(file="images/bg.png")
# Create label
main_label = Label(root, image=bg)
second_label = Label(root, image=bg)


def main_window():
    def resize(e):
        # adjust font size to window size
        # print(e)
        if e.width > 1600:
            banner_text.config(font=("SimSun", int(50)))
            take_photo_button.config(font=("Segoe UI", int(50)))
        elif 1400 >= e.width > 900:
            banner_text.config(font=("SimSun", int(30)))
            take_photo_button.config(font=("Segoe UI", int(30)))

    main_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Create text
    banner_text = Label(main_label, text="You don't know what breed you dog is? Check it out now!",
                        font=("SimSun", 50), fg="black", bg="#f4ead9")
    banner_text.place(relx=0.5, rely=0.1, anchor="center")

    # Create button
    take_photo_button = Button(main_label, text="Insert a picture of your dog.", font=("Segoe UI", 50), bg='#fffff5',
                               activebackground='#fffff5', command=move_to_second_label)
    take_photo_button.place(relx=0.5, rely=0.75, anchor="center")

    root.bind('<Configure>', resize)
    main_label.pack()
    root.mainloop()


def move_to_second_label():
    # Take image form user class
    z = img.upload_image()
    if z is not None:
        main_label.pack_forget()
        second_label.pack()
        second_window(z)


def move_to_first_label():
    second_label.pack_forget()
    main_label.pack()
    main_window()


def second_window(im):
    def resize(e):
        # adjust font size to window size
        # print(e)
        if e.width > 1600:
            b_text.config(font=("SimSun", int(50)))
            a_text.config(font=("Segoe UI", int(50)))
            check_button.config(font=("Segoe UI", int(50)))
            back_button.config(font=("Segoe UI", int(50)))
        elif 1400 >= e.width > 900:
            b_text.config(font=("SimSun", int(30)))
            a_text.config(font=("Segoe UI", int(30)))
            check_button.config(font=("Segoe UI", int(30)))
            back_button.config(font=("Segoe UI", int(30)))

    second_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Create text
    b_text = Label(second_label, text="Your dog is ...",
                   font=("SimSun", 50), fg="black", bg="#faf0de")
    b_text.place(relx=0.75, rely=0.1, anchor="center")

    # Place for image
    image_holder = Label(second_label, image=im, height=330, width=330)
    image_holder.image = im
    image_holder.place(relx=0.2, rely=0.5, anchor="center")

    # Text attribute - a place for the answer from the algorithm
    a_text = Label(second_label, text="",
                    font=("SimSun", 50), fg="black", bg="#fdf3e1")
    a_text.place(relx=0.60, rely=0.5, anchor="w")

    # Check button
    check_button = Button(second_label, text="Check it out.", font=("Segoe UI", 50), bg='#fffff5',
                          activebackground='#fffff5')
    check_button.place(relx=0.71, rely=0.79, anchor="center")
    # Back Button
    back_button = Button(second_label, text="Back", font=("Segoe UI", 40), bg='#fffff5',
                         activebackground='#fffff5', command=move_to_first_label)
    back_button.place(relx=0.1, rely=0.07, anchor="center")
    root.bind('<Configure>', resize)
    second_label.pack()


