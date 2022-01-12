import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from keras import models
from PIL import *
from keras.preprocessing import image
import numpy as np

color=1

def open_img():
    filename = filedialog.askopenfilename(title='open')
    img = Image.open(filename)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel['image']=img
    panel.image = img
    
    
    
    # File path
    filepath = 'modelLast.h5'

    # Load the model
    model = models.load_model(filepath, compile = True)

    image_path = filename
    new_img = image.load_img(image_path, target_size=(244, 244))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)

    # Generate predictions for samples
    predictions = model.predict(img)
    print(predictions)
    val=np.argmax(predictions)
    if val==0:
        result['text']="          COVID"
        result['fg']="red"
    elif val==1:
        result['text']="         NORMAL"
        result['fg']="green"
    else:
        result['text']="VIRAL PNEUMONIA"
        result['fg']="red"
    
def update():
    global color
    if(color%2==0):
        
        main['bg']="#432D50"
        button['fg']="white"
    else:
        main['bg']="#432D57"
        button['fg']="orange"
    color+=1
    main.after(800,update) 

main=tk.Tk()
main.geometry("450x700")
main.config(bg="#432D57")
main.title("COVID and VIRAL PNEUMONIA Detection App")

label=tk.Label(main,text="Test etmek istediğiniz resmi yükleyiniz.",bg="#432D57",fg="white",font=("times new roman",16,"italic"))
label.place(x=60,y=50)

button=tk.Button(main,text="Resim Yükle",bg="#432D51",fg="white",font=("times new roman",16), command=open_img,width=10,height=2)
button.place(x=145,y=100)
img = Image.open("Interface_img/logo.png")
img = img.resize((250, 250), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = Label(main, image=img)
panel.image = img
panel.place(x=90,y=200)

result=tk.Label(main,text="",bg="#432D57",fg="white",font=("times new roman",15,"bold"))
result.place(x=130,y=480)


logo_img = Image.open("Interface_img/logo3.png")
logo_img = logo_img.resize((100, 100))
logo_img = ImageTk.PhotoImage(logo_img)
logo = Label(main, image=logo_img,bd=0)
logo.image = logo_img
logo.place(x=350,y=600)
update() 

main.mainloop()










