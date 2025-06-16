# 🐶 Dog Image Classifier

A Python-based image classification project that utilizes pretrained convolutional neural networks (CNNs) to identify whether an image contains a dog, and if so, determine the dog breed. Built for evaluating model performance across architectures like **VGG**, **ResNet**, and **AlexNet**.

---

## 🚀 Features

- ✅ Detects whether an image contains a **dog** or **not a dog**
- 🐕 Identifies **dog breeds** using pretrained CNNs from PyTorch
- 📊 Calculates accuracy metrics for:
  - Dog detection
  - Breed classification
  - Not-a-dog filtering
- 💡 Supports image formats like `.jpg`, `.jpeg`, `.png`, and skips unsupported ones
- 🧠 Clean modular Python code with CLI support and results output to file

---

## 📁 Project Structure

```

.
├── pet\_images/           # Folder containing test images
├── check\_images.py       # Main program to execute the classification pipeline
├── classifier.py         # Pretrained CNN-based classification function
├── dognames.txt          # File containing valid dog breed names
├── results.txt           # Output file with classification results
├── README.md             # This documentation file
└── requirements.txt      # Project dependencies (optional)

````

---

## 🧪 Usage

### 🛠 Run the Classifier

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
````

### CLI Arguments

| Argument    | Description                                         | Default        |
| ----------- | --------------------------------------------------- | -------------- |
| `--dir`     | Path to the folder of pet images                    | `pet_images/`  |
| `--arch`    | CNN model architecture (`vgg`, `resnet`, `alexnet`) | `vgg`          |
| `--dogfile` | File with list of valid dog names                   | `dognames.txt` |

---

## ✅ Sample Output

```
📸 Total Images: 50
🐶 Dog Images: 42
🚫 Not-a-Dog Images: 8
✅ Correctly Classified Dogs: 40
🎯 Correct Dog Breeds: 33
❌ Correctly Classified 'Not-a-Dog': 8

🔢 Accuracy Metrics:
    % Correct Dogs:     95.2%
    % Correct Breeds:   78.6%
    % Correct NotDogs:  100.0%
```

Results are saved automatically to `results.txt`.

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/dog-image-classifier.git
cd dog-image-classifier

python -m venv venv
source venv/bin/activate      # or venv\\Scripts\\activate on Windows

pip install -r requirements.txt
```

> You can generate `requirements.txt` using:
>
> ```bash
> pip freeze > requirements.txt
> ```

---

## 🧠 Technologies Used

* Python 3
* PyTorch (`torch`, `torchvision`)
* PIL (Pillow)
* argparse
* Pretrained CNNs (ImageNet weights)

---

## 📄 License

MIT License © 2025 Your Name
Feel free to use and modify this project for learning and research purposes.

---

## ✨ Acknowledgements

* [PyTorch Models](https://pytorch.org/vision/stable/models.html)
* [ImageNet Dataset](http://www.image-net.org/)
* Udacity AI Programming with Python Nanodegree (Inspiration)

