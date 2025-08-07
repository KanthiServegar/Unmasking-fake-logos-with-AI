# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import load_model
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk

# # 1. Load CSV
# csv_path = "file_mapping.csv"
# df = pd.read_csv(csv_path)

# # 2. Parameters
# img_size = (100, 100)

# # 3. Load and preprocess images
# images = []
# labels = []

# for i, row in df.iterrows():
#     img_path = row['Filename']
#     label = 1 if row['Label'].lower() == 'genuine' else 0
#     if os.path.exists(img_path):
#         img = load_img(img_path, target_size=img_size)
#         img_array = img_to_array(img) / 255.0
#         images.append(img_array)
#         labels.append(label)

# X = np.array(images)
# y = to_categorical(labels, num_classes=2)

# # 4. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Build model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')  # 2 classes: Fake or Genuine
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# # 6. Save model
# model.save("model.h5")

# # 7. GUI for prediction
# def predict_image():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         img = load_img(file_path, target_size=img_size)
#         img_array = img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         prediction = model.predict(img_array)[0]
#         label = "Genuine" if np.argmax(prediction) == 1 else "Fake"
#         messagebox.showinfo("Prediction Result", f"The uploaded logo is: {label}")

#         # Display Image
#         img_disp = Image.open(file_path)
#         img_disp = img_disp.resize((200, 200))
#         img_tk = ImageTk.PhotoImage(img_disp)
#         panel.config(image=img_tk)
#         panel.image = img_tk

# # 8. Tkinter GUI
# root = tk.Tk()
# root.title("Fake Logo Detector")

# panel = tk.Label(root)
# panel.pack()

# btn = tk.Button(root, text="Upload Logo Image", command=predict_image)
# btn.pack(pady=20)

# root.mainloop()


# All your imports remain the same



from tkinter import *
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Global variables
main = Tk()
main.title("Fake Logo Detection")
main.geometry("1300x1200")
main.config(bg='burlywood2')

global filename, classifier, labels, X, Y, X_train, y_train, X_test, y_test
labels, X, Y = [], [], []

def getID(name):
    if name not in labels:
        labels.append(name)
    return labels.index(name)

def readLabels(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)

def uploadDataset():
    global filename
    labels.clear()
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, f"{filename} loaded\n\n")
    readLabels(filename)
    text.insert(END, "Logos found in dataset:\n\n")
    for label in labels:
        text.insert(END, label + "\n")

def processDataset():
    text.delete('1.0', END)
    global X, Y, X_train, y_train, X_test, y_test

    if not os.path.exists("model"):
        os.makedirs("model")

    if os.path.exists("model/X.txt.npy") and os.path.exists("model/Y.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X, Y = [], []
        for root, _, files in os.walk(filename):
            for file in files:
                if 'Thumbs.db' not in file:
                    img = cv2.imread(os.path.join(root, file))
                    img = cv2.resize(img, (64, 64))
                    X.append(img)
                    Y.append(getID(os.path.basename(root)))
        X = np.array(X, dtype='float32') / 255.0
        Y = np.array(Y)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)

    text.insert(END, f"Dataset Preprocessing Completed\nTotal images: {X.shape[0]}\n\n")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y_cat = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2)
    text.insert(END, f"80% training samples: {X_train.shape[0]}\n20% testing samples: {X_test.shape[0]}\n")

def trainCNN():
    text.delete('1.0', END)
    global classifier

    classifier = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.exists("model/model_weights.keras"):
        checkpoint = ModelCheckpoint("model/model_weights.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
        hist = classifier.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=1)
        with open("model/history.pckl", "wb") as f:
            pickle.dump(hist.history, f)
    else:
        classifier = load_model("model/model_weights.keras")

    preds = classifier.predict(X)
    y_true = np.argmax(Y, axis=1)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

    text.insert(END, f"CNN Accuracy  : {acc:.2f}\nCNN Precision : {prec:.2f}\nCNN Recall    : {rec:.2f}\nCNN F1 Score  : {f1:.2f}\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title("CNN Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def graph():
    if not os.path.exists("model/history.pckl"):
        messagebox.showerror("Error", "No training history found!")
        return
    with open("model/history.pckl", "rb") as f:
        hist = pickle.load(f)

    acc = hist['val_accuracy']
    loss = hist['val_loss']
    plt.figure(figsize=(10,6))
    plt.plot(acc, 'g-', label='Validation Accuracy')
    plt.plot(loss, 'r-', label='Validation Loss')
    plt.title("CNN Accuracy & Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def classifyLogo():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please train or load model first!")
        return
    testfile = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(testfile)
    resized = cv2.resize(img, (64, 64)).astype('float32') / 255.0
    pred = classifier.predict(np.expand_dims(resized, axis=0))
    label = labels[np.argmax(pred)]

    show_img = cv2.resize(img, (700, 400))
    cv2.putText(show_img, 'Predicted: ' + label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow("Logo Prediction", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectAndClassifyMultipleLogos():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please train or load the model first!")
        return

    testfile = filedialog.askopenfilename(initialdir="testImages")
    if not testfile:
        return

    img = cv2.imread(testfile)
    img = cv2.resize(img, (700, int(img.shape[0] * 700 / img.shape[1])))  # Resize width to 700
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Better thresholding method
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        # Filter out very small or very large areas
        if area > 2000 and w > 50 and h > 50 and 0.5 < w/h < 2.5:
            roi = img[y:y+h, x:x+w]
            try:
                resized = cv2.resize(roi, (64, 64)).astype('float32') / 255.0
                pred = classifier.predict(np.expand_dims(resized, axis=0))[0]
                label = labels[np.argmax(pred)]
                color = (0, 255, 0) if label.lower() in ['genuine', 'real'] else (0, 0, 255)

                cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except:
                pass

    cv2.imshow("Detected & Classified Logos", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np
from tkinter import filedialog, messagebox

def detectAndClassifyLogosInVideo():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please train or load the model first!")
        return

    videopath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not videopath:
        return

    cap = cv2.VideoCapture(videopath)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (700, int(frame.shape[0] * 700 / frame.shape[1])))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 3)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if area > 2000 and w > 50 and h > 50 and 0.5 < w/h < 2.5:
                roi = img[y:y+h, x:x+w]
                try:
                    resized = cv2.resize(roi, (64, 64)).astype('float32') / 255.0
                    pred = classifier.predict(np.expand_dims(resized, axis=0))[0]
                    label = labels[np.argmax(pred)]
                    color = (0, 255, 0) if label.lower() in ['genuine', 'real'] else (0, 0, 255)

                    cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except:
                    pass

        cv2.imshow("Video - Detected Logos", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def close():
    main.destroy()

# GUI Widgets
font = ('times', 16, 'bold')
Label(main, text='Fake Logo Detection', bg='LightGoldenrod1', fg='medium orchid', font=font, height=3, width=120).place(x=0, y=5)

font1 = ('times', 13, 'bold')
Button(main, text="Upload Logo Dataset", command=uploadDataset, font=font1).place(x=50, y=100)
Button(main, text="Preprocess Dataset", command=processDataset, font=font1).place(x=50, y=150)
Button(main, text="Train CNN Algorithm", command=trainCNN, font=font1).place(x=50, y=200)
Button(main, text="CNN Training Graph", command=graph, font=font1).place(x=50, y=250)
Button(main, text="Logo Classification", command=classifyLogo, font=font1).place(x=50, y=300)
Button(main, text="Detect & Classify Multiple Logos", command=detectAndClassifyMultipleLogos, font=font1).place(x=50, y=350)


Button(main, text="Detect Logos in Video", command=detectAndClassifyLogosInVideo, font=font1).place(x=50, y=400)


Button(main, text="Exit", command=close, font=font1).place(x=50, y=450)

pathlabel = Label(main, bg='yellow4', fg='white', font=font1)
pathlabel.place(x=50, y=500)

text = Text(main, height=25, width=78, font=('times', 12, 'bold'))
text.place(x=370, y=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)

main.mainloop()