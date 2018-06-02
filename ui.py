from tkinter import Tk, Label, Button, filedialog
from tkinter import messagebox
from tkinter import *
import os
from shutil import copyfile
import random
import numpy as np
import model_obj
import test_image
import cv2

class GUI:
    def __init__(self):
        self.master = Tk()
        self.master.title('NeuroStyle')
        self.master.resizable(False, False)

        self.model_filename = ''
        self.epochs = IntVar()
        self.epochs.set(10)
        self.batchsize = IntVar()
        self.batchsize.set(32)
        self.view_on_create = BooleanVar()
        self.view_on_create.set(False)
        self.img_output_size = IntVar()
        self.img_output_size.set(600)
        self.model_filename_create = ''
        self.todo_list = []
        self.picture_dict = {}
        self.pixels_per_img = IntVar()
        self.pixels_per_img.set(5000)
        self.subsection_radius = IntVar()
        self.subsection_radius.set(4)

        self.header_label = Label(self.master, text='NeuroStyle', pady=10, padx=20, font=('Consolas', 30))
        self.header_label.pack()

        self.upper_frame = Frame(self.master)
        self.upper_frame.pack(side='top', pady=10)

        self.model_selection_frame = Frame(self.upper_frame, padx=10)
        self.model_selection_frame.pack(side='left')

        self.model_frame = Frame(self.model_selection_frame)
        self.model_frame.pack(side='top', pady=4)

        self.model_label = Label(self.model_frame, text='Model:', width=15)
        self.model_label.pack(side='left')
        self.model_select_button = Button(self.model_frame, text='No model selected', command=self.selectModel, width=23, pady=10)
        self.model_select_button.pack(side='left')

        self.model_train_frame = Frame(self.model_selection_frame)
        self.model_train_frame.pack(side='top', pady=4)

        self.picture_label = Label(self.model_train_frame, text='Training Pictures:', width=15)
        self.picture_label.pack(side='left')
        self.add_button = Button(self.model_train_frame, text='Add', command=self.addTrainingPicture, width=7, pady=10)
        self.add_button.pack(side='left')
        self.delete_button = Button(self.model_train_frame, text='Delete', command=self.deleteTrainingPicture, width=7, pady=10)
        self.delete_button.pack(side='left')
        self.train_button = Button(self.model_train_frame, text='Train', command=self.trainModel, width=7, pady=10)
        self.train_button.pack(side='left')

        self.model_save_frame = Frame(self.model_selection_frame)
        self.model_save_frame.pack(side='left')

        self.model_save_label = Label(self.model_save_frame, text='Model Options', width=15)
        self.model_save_label.pack(side='left')
        self.model_saveas_button = Button(self.model_save_frame, text='Save As', command=self.modelSaveas, width=11, pady=10)
        self.model_saveas_button.pack(side='left')
        self.model_new_button = Button(self.model_save_frame, text='New', command=self.modelNew, width=12, pady=10)
        self.model_new_button.pack(side='top')

        self.training_settings_frame = Frame(self.upper_frame, padx=10)
        self.training_settings_frame.pack(side='top')

        self.training_settings_label = Label(self.training_settings_frame, text='Training Settings', font=('Consolas', 15))
        self.training_settings_label.pack(side='top')
        self.epoch_setting_frame = Frame(self.training_settings_frame, pady=10)
        self.epoch_setting_frame.pack(side='top')
        self.epoch_setting_label = Label(self.epoch_setting_frame, text='Epochs:', width=15)
        self.epoch_setting_label.pack(side='left')
        self.epoch_setting_select = Spinbox(self.epoch_setting_frame, from_=1, to=1000, textvariable=self.epochs, width=10)
        self.epoch_setting_select.pack(side='left')
        self.batchsize_setting_frame = Frame(self.training_settings_frame)
        self.batchsize_setting_frame.pack(side='top')
        self.batchsize_setting_label = Label(self.batchsize_setting_frame, text='Batch Size:', width=15)
        self.batchsize_setting_label.pack(side='left')
        self.batchsize_setting_select = Spinbox(self.batchsize_setting_frame, from_=1, to=10000, textvariable=self.batchsize, width=10)
        self.batchsize_setting_select.pack(side='left')
        self.pixels_setting_frame = Frame(self.training_settings_frame, pady=10)
        self.pixels_setting_frame.pack(side='top')
        self.pixels_setting_label = Label(self.pixels_setting_frame, text='Pixels Per Image:', width=15)
        self.pixels_setting_label.pack(side='left')
        self.pixels_setting_select = Spinbox(self.pixels_setting_frame, from_=10, to=10000, textvariable=self.pixels_per_img, width=10)
        self.pixels_setting_select.pack(side='left')
        self.radius_setting_frame = Frame(self.training_settings_frame)
        self.radius_setting_frame.pack(side='top')
        self.radius_setting_label = Label(self.radius_setting_frame, text='Subsection Radius:', width=15)
        self.radius_setting_label.pack(side='left')
        self.radius_setting_select = Spinbox(self.radius_setting_frame, from_=1, to=10, textvariable=self.subsection_radius, width=10)
        self.radius_setting_select.pack(side='left')

        self.view_header_frame = Frame(self.master, pady=20)
        self.view_header_frame.pack(side='top')
        self.view_header_label = Label(self.view_header_frame, text='Viewing', font=('Consolas', 20), relief=RIDGE, width=35)
        self.view_header_label.pack(side='top')

        self.creating_frame = Frame(self.master, pady=10)
        self.creating_frame.pack(side='top')

        self.to_create_frame = Frame(self.creating_frame, padx=10)
        self.to_create_frame.pack(side='left')
        self.to_create_label = Label(self.to_create_frame, text='To Create', font=('Consolas', 12))
        self.to_create_label.pack(side='top')
        self.to_create_combobox_frame = Frame(self.to_create_frame)
        self.to_create_combobox_frame.pack(side='top')
        self.to_create_scrollbar = Scrollbar(self.to_create_combobox_frame)
        self.to_create_scrollbar.pack(side='right')
        self.to_create_listbox = Listbox(self.to_create_combobox_frame, yscrollcommand=self.to_create_scrollbar.set)
        self.to_create_listbox.pack(side='left')
        self.to_create_button_frame = Frame(self.to_create_frame)
        self.to_create_button_frame.pack(side='top')
        self.to_create_add_button = Button(self.to_create_button_frame, text='Add', command=self.addToDo, pady=10, width=10)
        self.to_create_add_button.pack(side='left')
        self.to_create_delete_button = Button(self.to_create_button_frame, text='Delete', command=self.deleteToDo, pady=10, width=10)
        self.to_create_delete_button.pack(side='left')
        self.to_create_view_check = Checkbutton(self.to_create_frame, text='View on create?', var=self.view_on_create, pady=10)
        self.to_create_view_check.pack(side='top')

        self.creating_settings_frame = Frame(self.creating_frame, padx=10)
        self.creating_settings_frame.pack(side='top')
        self.creating_settings_label = Label(self.creating_settings_frame, text='Creation Settings', font=('consolas', 15))
        self.creating_settings_label.pack(side='top')
        self.img_output_size_frame = Frame(self.creating_settings_frame)
        self.img_output_size_frame.pack(side='top')
        self.img_output_size_label = Label(self.img_output_size_frame, text='Image Output Size:', pady=10, width=20)
        self.img_output_size_label.pack(side='left')
        self.img_output_size_select = Spinbox(self.img_output_size_frame, from_=100, to=4000, textvariable=self.img_output_size, width=15)
        self.img_output_size_select.pack(side='left')
        self.model_select_create_frame = Frame(self.creating_settings_frame)
        self.model_select_create_frame.pack(side='top')
        self.model_select_create_label = Label(self.model_select_create_frame, text='Model', width=20)
        self.model_select_create_label.pack(side='left')
        self.model_select_create_button = Button(self.model_select_create_frame, text='No model selected', command=self.selectCreateModel, width=14, pady=10)
        self.model_select_create_button.pack(side='left')

        self.start_queue_frame = Frame(self.creating_frame, padx=10, pady=60)
        self.start_queue_frame.pack(side='top')
        self.start_queue_button = Button(self.start_queue_frame, text='Start Creation Queue', command=self.startQueue, padx=10, pady=27)
        self.start_queue_button.pack(side='top')

    def startQueue(self):
        if len(self.todo_list) == 0:
            messagebox.showerror('Error', 'To do list is empty')
            return
        if self.model_filename_create == '':
            messagebox.showerror('Error', 'Model not selected')
            return
        estimator = model_obj.SketchModel(None)
        estimator.loadModel(self.model_filename_create)
        for picture in self.todo_list:
            produce_image = test_image.TestImage(size=self.img_output_size, subsection_radius=self.subsection_radius)
            produce_image.openImage(picture)
            final_img = produce_image.getPrediction(estimator.model)
            final_img *= 255.0
            final_img = np.absolute(final_img)
            final_img = final_img.astype(dtype=np.uint8)
            cv2.imwrite('./saved_images/' + os.path.basename(picture)[:-4] + '.PNG', final_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if self.view_on_create.get():
                cv2.imshow('Produced Image', final_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def updateToDo(self):
        self.to_create_listbox.delete(0, END)
        for filename in self.todo_list:
            filename = os.path.basename(filename)
            self.to_create_listbox.insert(END, filename)

    def selectCreateModel(self):
        model_filename_create = filedialog.askopenfilename(initialdir='./models')
        try:
            filename, file_extension = os.path.splitext(model_filename_create)
            if filename == '':
                messagebox.showwarning('Warning', 'Model not selected.')
            elif file_extension != '.h5':
                messagebox.showerror('Error', 'The file selected did not have extension ".h5"')
            else:
                self.model_select_create_button['text'] = os.path.basename(model_filename_create)
                self.model_filename_create = model_filename_create
        except:
            messagebox.showerror('Error', 'Error selecting file.')

    def addToDo(self):
        picture_filename = filedialog.askopenfilename()
        try:
            filename, file_extension = os.path.splitext(picture_filename)
            if filename == '':
                messagebox.showwarning('Warning', 'Picture not selected.')
            elif file_extension != '.PNG' and file_extension != '.jpg' and file_extension != '.png':
                messagebox.showerror('Error', 'The file selected did not have extension ".PNG" or ".jpg"')
            else:
                self.todo_list.append(filename + file_extension)
                self.updateToDo()
        except:
            messagebox.showerror('Error', 'Error selecting file.')

    def deleteToDo(self):
        selection = self.to_create_listbox.curselection()
        if selection != ():
            selection = selection[0]
            del self.todo_list[selection]
            self.updateToDo()
        else:
            if len(self.todo_list) != 0:
                del self.todo_list[-1]
                self.updateToDo()

    def modelSaveas(self):
        if self.model_filename == '':
            messagebox.showerror('Error', 'Model not selected')
            return
        filename = filedialog.asksaveasfile(mode='w', defaultextension='.h5')
        copyfile(self.model_filename, filename.name)

    def submitName(self):
        self.new_name = self.entry_bonus.get()
        self.popupEntry.destroy()
        if self.new_name != '':
            existing_models = os.listdir('./models')
            if self.new_name + '.h5' in existing_models:
                messagebox.showerror('Error', 'Model already exists')
                return
            file = open('./models/' + self.new_name + '.h5', 'w')
            file.close()
        else:
            messagebox.showwarning('Warning', 'Model name not chosen.')

    def modelNew(self):
        self.new_name = ''
        self.popupEntry = Toplevel()
        self.popupEntry.title('Enter Name')
        self.label_bonus = Label(self.popupEntry, text='Name of new model', width=30, pady=10)
        self.label_bonus.pack(side='top')
        self.entry_bonus = Entry(self.popupEntry, width=30)
        self.entry_bonus.pack(side='top')
        self.submit_bonus = Button(self.popupEntry, text='Submit', command=self.submitName, width=15, pady=10)
        self.submit_bonus.pack(side='top')
        self.popupEntry.mainloop()

    def trainModel(self):
        if self.model_filename == '':
            messagebox.showerror('Error', 'No model selected')
        elif len(self.picture_dict) == 0:
            messagebox.showerror('Error', 'No training data provided')
        else:
            training_data, truth_data = [], []
            for filename in self.picture_dict:
                original = test_image.TestImage(subsection_radius=self.subsection_radius)
                original.openImageTraining(filename)
                sketch = test_image.TestImage(subsection_radius=self.subsection_radius)
                sketch.openImageTraining(self.picture_dict[filename])
                img_size = (original.image.shape[1], original.image.shape[0])
                other_img_size = (sketch.image.shape[1], sketch.image.shape[0])
                if img_size != other_img_size:
                    messagebox.showerror('Training set contains pair whose sizes do not match: ' +
                                         os.path.basename(filename) + ' and ' +
                                         os.path.basename(self.picture_dict[filename]))
                    return

                for itr in range(self.pixels_per_img.get()):
                    x = random.randint(0, original.image.shape[1]-1)
                    y = random.randint(0, original.image.shape[0]-1)
                    training_data.append(original.getPixelSection(x, y))
                    truth_data.append(sketch.getPixel(x, y))
            training_data = np.asarray(training_data, dtype=np.float32)
            truth_data = np.asarray(truth_data, dtype=np.float32)

            estimator = model_obj.SketchModel(training_data.shape)
            try:
                estimator.loadModel(self.model_filename)
            except:
                messagebox.showerror('Error', 'Error opening model')
                return
            estimator.trainModel(training_data, truth_data, self.batchsize.get(), self.epochs.get())
            estimator.saveModel(self.model_filename)

    def deleteTrainingPicture(self):
        training_picture = filedialog.askopenfilename(initialdir='./training')
        if training_picture == '':
            messagebox.showwarning('Warning', 'Picture not selected')
        elif training_picture not in self.picture_dict.keys():
            messagebox.showerror('Error', 'That picture is not in the training set')
        else:
            os.remove(training_picture)
            os.remove(self.picture_dict[training_picture])
            del self.picture_dict[training_picture]

    def addTrainingPicture(self):
        training_picture = filedialog.askopenfilename(initialdir='./unused')
        try:
            filename, file_extension = os.path.splitext(training_picture)
            if filename == '':
                messagebox.showwarning('Warning', 'Picture not selected.')
            elif file_extension != '.PNG' and file_extension != '.jpg' and file_extension != '.png':
                messagebox.showerror('Error', 'The file selected did not have extension ".PNG" or ".jpg"')
            else:
                truth_picture = filedialog.askopenfilename(initialdir='./unused')
                try:
                    filename, file_extension = os.path.splitext(truth_picture)
                    if filename == '':
                        messagebox.showwarning('Warning', 'Picture not selected.')
                    elif file_extension != '.PNG' and file_extension != '.jpg' and file_extension != '.png':
                        messagebox.showerror('Error', 'The file selected did not have extension ".PNG" or ".jpg"')
                    else:
                        training_path = (os.getcwd() + '\\training\\' + os.path.basename(training_picture)).replace('\\', '/')
                        truth_path = (os.getcwd() + '\\truth\\' + os.path.basename(truth_picture)).replace('\\', '/')
                        copyfile(training_picture, './training/' + os.path.basename(training_picture))
                        copyfile(truth_picture, './truth/' + os.path.basename(truth_picture))
                        self.picture_dict[training_path] = truth_path
                except:
                    messagebox.showerror('Error', 'Error selecting pictures')
        except:
            messagebox.showerror('Error', 'Error selecting pictures')

    def selectModel(self):
        model_filename = filedialog.askopenfilename(initialdir='./models')
        try:
            filename, file_extension = os.path.splitext(model_filename)
            if filename == '':
                messagebox.showwarning('Warning', 'Model not selected.')
            elif file_extension != '.h5':
                messagebox.showerror('Error', 'The file selected did not have extension ".h5"')
            else:
                self.model_select_button['text'] = os.path.basename(model_filename)
                self.model_filename = model_filename
        except:
            messagebox.showerror('Error', 'Error selecting file.')

gui = GUI()
gui.master.mainloop()
for file in os.listdir('./training'):
    os.remove(os.getcwd() + './training/' + file)
for file in os.listdir('./truth'):
    os.remove(os.getcwd() + './truth/' + file)