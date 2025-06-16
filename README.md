
# 🐶 Dog Image Classifier

A Python-based image classification project that uses pretrained convolutional neural networks (CNNs) to identify whether an image contains a dog, and if so, determine its breed. This project supports evaluating three models: **VGG**, **ResNet**, and **AlexNet**.

---

## 🚀 Features

- ✅ Detects whether an image contains a **dog** or not.
- 🐕 Identifies **dog breeds** using pretrained CNNs from PyTorch.
- 📊 Computes accuracy metrics for:
  - Dog detection
  - Breed classification
  - Non-dog filtering
- 🖼️ Skips unsupported image formats automatically.
- 📝 Outputs detailed classification results to a file.

---

## 🧠 Technologies Used

- Python 3
- PyTorch (`torch`, `torchvision`)
- Pillow (PIL)
- argparse

---

## 📁 Project Structure

```

dog-image-classifier/
├── pet\_images/              # Folder containing test images
├── classifier.py            # CNN-based image classification function
├── check\_images.py          # Main program: runs full classification pipeline
├── dognames.txt             # List of valid dog breed names
├── results.txt              # Output file containing results
├── requirements.txt         # Required dependencies
└── README.md                # Project documentation

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/dog-image-classifier.git
cd dog-image-classifier
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 How to Run

Make sure your images are inside the `pet_images/` folder and each image filename contains the true label (e.g., `golden_retriever_01.jpg`).

Then run:

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

### CLI Arguments

| Argument    | Description                                         | Default        |
| ----------- | --------------------------------------------------- | -------------- |
| `--dir`     | Path to the folder of pet images                    | `pet_images/`  |
| `--arch`    | CNN model architecture (`vgg`, `resnet`, `alexnet`) | `vgg`          |
| `--dogfile` | Text file containing valid dog names                | `dognames.txt` |

---

## 🔍 How Labels Are Compared

In this project, we evaluate the accuracy of pretrained image classifiers by comparing:

| Term                | Meaning                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **True Label**      | The real label of the image (e.g., `"golden retriever"`), extracted from the image filename.              |
| **Predicted Label** | The label predicted by the pretrained CNN (e.g., VGG, ResNet, AlexNet) after analyzing the image content. |

### 🧠 Why is the filename used for the True Label?

Since we don’t have a separate file containing ground truth labels, we use the **filename itself** to extract the actual class.
For example:

```
Filename: golden_retriever_01.jpg  
→ True label: "golden retriever"
```

This true label is then compared against the classifier’s predicted output.

---

### 🐾 What if the image contains more than one animal?

If an image contains a dog and another animal (like a cat), the classifier may detect **whichever is more visually dominant**.
The model returns only **one label** per image — it does not support multi-label classification. In such cases:

* If the true label is `"dog"` and prediction is `"cat"` → it's incorrect.
* If the true label is `"dog"` and prediction is `"dog"` → it's correct (regardless of the other animal).

---

## ✅ Sample Output

```
📈 Final Statistics Report
📸 Total Images: 50
🐶 Dog Images: 42
🚫 Not-a-Dog Images: 8
✅ Correctly Classified Dogs: 39
🎯 Correct Dog Breeds: 32
❌ Correctly Classified 'Not-a-Dog': 8

🔢 Accuracy Metrics:
    % Correct Dogs:     92.9%
    % Correct Breeds:   76.2%
    % Correct NotDogs:  100.0%
```

All results are saved automatically to `results.txt`.

---

## 📦 requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
```

To install:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License © 2025 Batool Shilleh
Feel free to use, fork, or contribute to the project.

---

## ✨ Acknowledgements

* [PyTorch Pretrained Models](https://pytorch.org/vision/stable/models.html)
* [ImageNet Dataset](http://www.image-net.org/)
* Udacity AI Programming with Python Nanodegree (Inspiration)
