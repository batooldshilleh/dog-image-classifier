import time
import argparse
import os
from classifier import classifier


def get_input_args():
    """
    Parses command line arguments for image directory, CNN architecture, and dog names file.
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', help='Path to folder of pet images')
    parser.add_argument('--arch', type=str, default='vgg', help='CNN model architecture: vgg, resnet, alexnet')
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help='Text file with dog breed names')
    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Extracts pet labels from image filenames.
    Returns:
        dict: Keys are filenames, values are labels (lowercase string)
    """
    results_dic = dict()
    for filename in os.listdir(image_dir):
        if filename.startswith('.'):
            continue
        name_parts = filename.lower().split("_")
        label = " ".join([word for word in name_parts if word.isalpha()]).strip()
        results_dic[filename] = label
    return results_dic


def classify_images(images_dir, pet_labels_dic, model):
    """
    Classifies pet images and compares results with true labels.
    Returns:
        dict: Keys are filenames, values are [true_label, predicted_label, match]
    """
    results_dic = {}
    for filename, true_label in pet_labels_dic.items():
        full_path = os.path.join(images_dir, filename)
        predicted_label = classifier(full_path, model)
        if predicted_label is None:
            continue
        predicted_label = predicted_label.lower().strip()
        match = 1 if true_label in predicted_label else 0
        results_dic[filename] = [true_label, predicted_label, match]
    return results_dic


def adjust_results4_isadog(results_dic, dogfile):
    """
    Adds dog/not-dog classification flags to results dictionary.
    Modifies:
        results_dic: Adds index 3 = is_dog, index 4 = classifier_is_dog
    """
    with open(dogfile, "r") as f:
        dog_names = set(name.strip().lower() for name in f.readlines())

    for key in results_dic:
        true_label = results_dic[key][0]
        classifier_label = results_dic[key][1]
        is_dog = 1 if true_label in dog_names else 0
        is_classifier_dog = int(any(name in classifier_label for name in dog_names))
        results_dic[key].extend([is_dog, is_classifier_dog])


def calculates_results_stats(results_dic):
    """
    Calculates accuracy statistics based on classification results.
    Returns:
        dict: Summary statistics
    """
    stats = {
        'n_images': len(results_dic),
        'n_dogs_img': 0,
        'n_notdogs_img': 0,
        'n_correct_dogs': 0,
        'n_correct_notdogs': 0,
        'n_correct_breed': 0,
    }
    for value in results_dic.values():
        is_dog, classifier_is_dog, match = value[3], value[4], value[2]
        if is_dog:
            stats['n_dogs_img'] += 1
            if classifier_is_dog:
                stats['n_correct_dogs'] += 1
            if match:
                stats['n_correct_breed'] += 1
        else:
            stats['n_notdogs_img'] += 1
            if not classifier_is_dog:
                stats['n_correct_notdogs'] += 1

    stats['pct_correct_dogs'] = (stats['n_correct_dogs'] / stats['n_dogs_img'] * 100) if stats['n_dogs_img'] else 0
    stats['pct_correct_breed'] = (stats['n_correct_breed'] / stats['n_dogs_img'] * 100) if stats['n_dogs_img'] else 0
    stats['pct_correct_notdogs'] = (stats['n_correct_notdogs'] / stats['n_notdogs_img'] * 100) if stats['n_notdogs_img'] else 0

    return stats


def print_classification_results(results_dic):
    print("\n📊 Classification Samples:")
    for filename, values in results_dic.items():
        print(f"{filename} → True: {values[0]} | Predicted: {values[1]} | Match: {values[2]} | IsDog: {values[3]} | PredIsDog: {values[4]}")


def print_summary(stats):
    print("\n📈 Final Statistics Report")
    print(f"📸 Total Images: {stats['n_images']}")
    print(f"🐶 Dog Images: {stats['n_dogs_img']}")
    print(f"🚫 Not-a-Dog Images: {stats['n_notdogs_img']}")
    print(f"✅ Correctly Classified Dogs: {stats['n_correct_dogs']}")
    print(f"🎯 Correct Dog Breeds: {stats['n_correct_breed']}")
    print(f"❌ Correctly Classified 'Not-a-Dog': {stats['n_correct_notdogs']}")
    print("\n🔢 Accuracy Metrics:")
    print(f"    % Correct Dogs:     {stats['pct_correct_dogs']:.1f}%")
    print(f"    % Correct Breeds:   {stats['pct_correct_breed']:.1f}%")
    print(f"    % Correct NotDogs:  {stats['pct_correct_notdogs']:.1f}%")


def save_results_to_file(results_dic, stats, filepath="results.txt"):
    with open(filepath, "w") as f:
        f.write("📈 Final Statistics Report\n")
        f.write(f"📸 Total Images: {stats['n_images']}\n")
        f.write(f"🐶 Dog Images: {stats['n_dogs_img']}\n")
        f.write(f"🚫 Not-a-Dog Images: {stats['n_notdogs_img']}\n")
        f.write(f"✅ Correctly Classified Dogs: {stats['n_correct_dogs']}\n")
        f.write(f"🎯 Correct Dog Breeds: {stats['n_correct_breed']}\n")
        f.write(f"❌ Correctly Classified 'Not-a-Dog': {stats['n_correct_notdogs']}\n")
        f.write("\n🔢 Accuracy Metrics:\n")
        f.write(f"    % Correct Dogs:     {stats['pct_correct_dogs']:.1f}%\n")
        f.write(f"    % Correct Breeds:   {stats['pct_correct_breed']:.1f}%\n")
        f.write(f"    % Correct NotDogs:  {stats['pct_correct_notdogs']:.1f}%\n")
        f.write("\n🖼️ Classification Samples:\n")
        for filename, values in results_dic.items():
            f.write(f"{filename} → True: {values[0]} | Predicted: {values[1]} | Match: {values[2]} | IsDog: {values[3]} | PredIsDog: {values[4]}\n")


def main():
    start_time = time.time()
    in_arg = get_input_args()
    pet_labels = get_pet_labels(in_arg.dir)
    results_dic = classify_images(in_arg.dir, pet_labels, in_arg.arch)
    adjust_results4_isadog(results_dic, in_arg.dogfile)
    stats = calculates_results_stats(results_dic)
    print_classification_results(results_dic)
    print_summary(stats)
    save_results_to_file(results_dic, stats)
    total_time = time.time() - start_time
    print(f"\n⏱️ Total Runtime: {int(total_time // 3600)}:{int((total_time % 3600) // 60):02d}:{int(total_time % 60):02d} (hh:mm:ss)")


if __name__ == "__main__":
    main()
