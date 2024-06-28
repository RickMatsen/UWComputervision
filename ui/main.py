from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter as tk
from tkinter.filedialog import asksaveasfilename,askopenfilename, askdirectory
from PIL import Image, ImageTk
from inference import *

MODEL = None
MODEL_PATH = "resnet34-600ep-adam"
FILE_PATH = None

def on_drop(event):
    """Handle the drop event for images."""
    global FILE_PATH
    FILE_PATH = event.data
    FILE_PATH = FILE_PATH.strip('{}')
    if FILE_PATH:
        open_image(FILE_PATH)

def open_image(filepath=None):
    """Open an image for viewing, resizing it to fit within a maximum size."""
    global img, photo  # Make img and photo accessible outside open_image
    max_width, max_height = 600, 400  # Maximum dimensions for the displayed image
    global FILE_PATH
    if not filepath:
        filepath = askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All Files", "*.*")]
        )
        FILE_PATH = filepath
        if not filepath:
            return
    img = Image.open(filepath)
    
    # Resize the image to fit within the maximum dimensions while maintaining aspect ratio
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    
    photo = ImageTk.PhotoImage(img)
    canvas.config(width=max_width, height=max_height)
    canvas.create_image(max_width/2, max_height/2, image=photo, anchor=tk.CENTER)
    window.title(f"Image Viewer - {filepath}")

def rotate_image():
    """Rotate the current image, maintaining the display size."""
    global img, photo
    if img is None:  # Check if an image has been loaded
        return
    img = img.rotate(90, expand=True)
    # After rotating, resize the image to fit the canvas if necessary
    max_width, max_height = 600, 400  # Keep the same maximum dimensions
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    
    photo = ImageTk.PhotoImage(img)
    canvas.config(width=max_width, height=max_height)
    canvas.create_image(max_width/2, max_height/2, image=photo, anchor=tk.CENTER)

def annotate():
    """Annotate the current image"""
    global MODEL
    if not MODEL:
        MODEL = load_model(MODEL_PATH)


    global img, photo, FILE_PATH
    if FILE_PATH is None:  # Check if an image has been loaded
        return
    img = run_inference(MODEL, FILE_PATH)
    max_width, max_height = 600, 400  # Keep the same maximum dimensions
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    
    photo = ImageTk.PhotoImage(img)
    canvas.config(width=max_width, height=max_height)
    canvas.create_image(max_width/2, max_height/2, image=photo, anchor=tk.CENTER)

def save_image():
    """Save the current image."""
    if img is None:
        return
    filepath = asksaveasfilename(
        defaultextension="*.png",
        filetypes=[("PNG Files", "*.png"), ("JPG Files", "*.jpg"), ("All Files", "*.*")],
    )
    if not filepath:
        return
    img.save(filepath)

def update_canvas_with_paths(titles, paths):
    """Update the canvas with the provided paths."""
    canvas.delete("all")  # Clear the canvas
    start_y = 20
    for title, path in zip(titles, paths):
        canvas.create_text(10, start_y, text=f"{title}:{path}", anchor='nw', font=('Helvetica', 10, 'normal'))
        start_y += 20  # Increment y position for the next text
    canvas.create_text(10, start_y, text="Processing...", anchor='nw', font=('Helvetica', 10, 'normal'))
    infer_annot(*paths)
    start_y += 20  # Increment y position for the next text
    canvas.create_text(10, start_y, text="Done", anchor='nw', font=('Helvetica', 10, 'normal'))


    

def get_file_paths():
    """Prompt the user to select three file paths and display them on the canvas."""
    paths = []
    titles = ["data path", "model path", "annotation path"]
    for title in titles:
        if title == "data path":
            path = askdirectory(title=title)
        else:
            path = askopenfilename(title=title, filetypes=[("All Files", "*.*")])
        if not path:
            tk.messagebox.showinfo("Operation Cancelled", f"You did not select a file for {title}.")
            return
        paths.append(path)

    # If all three paths are successfully selected, update the canvas
    if len(paths) == 3:
        update_canvas_with_paths(titles, paths)



# Use TkinterDnD.Tk() instead of tk.Tk()
window = TkinterDnD.Tk()
window.title("Image Viewer")

img = None

window.rowconfigure(0, minsize=400, weight=1)
window.columnconfigure(1, minsize=600, weight=1)

canvas = tk.Canvas(window, width=600, height=400)  # Set initial canvas size
frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_open = tk.Button(frm_buttons, text="Open Image", command=lambda: open_image())
btn_rotate = tk.Button(frm_buttons, text="Rotate Image", command=rotate_image)
btn_save = tk.Button(frm_buttons, text="Save Image", command=save_image)
btn_annotate = tk.Button(frm_buttons, text="Annotate", command=annotate)
btn_get_paths = tk.Button(frm_buttons, text="Get File Paths", command=get_file_paths)
btn_get_paths.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_rotate.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
btn_annotate.grid(row=2, column=0, sticky="ew",padx=5, pady=5 )
btn_save.grid(row=3, column=0, sticky="ew", padx=5, pady=5)


frm_buttons.grid(row=0, column=0, sticky="ns")
canvas.grid(row=0, column=1, sticky="nsew")

# Register window as a drop target
window.drop_target_register(DND_FILES)
window.dnd_bind('<<Drop>>', on_drop)

window.mainloop()
