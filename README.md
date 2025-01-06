# Skin Cancer Detection using CNN and Machine Learning

## Project Overview
This project aims to assist dermatologists in detecting and classifying different types of skin cancers using Convolutional Neural Networks (CNN) and Machine Learning techniques. The system is capable of identifying the following skin conditions:

- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

## Dataset
The dataset used in this project is sourced from Kaggle, uploaded by ISI (International Skin Imaging Centre). It contains approximately 2000 dermoscopic images across 9 skin cancer classes.

- Dataset Link: [Skin Cancer Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

## Models
The project includes five different models with varying layer structures and complexities:
- **3 CNN Models**
- **2 ResNet-50 Models**

Each model provides independent predictions, allowing comparison and cross-validation of results.

## How it Works
1. The doctor uploads a dermoscopic image.
2. Each of the five models processes the image and provides a prediction.
3. The results from all models are displayed for evaluation.

## Steps to Execute
1. Download the files from this repository into a specific folder.
2. Download the dataset from the provided Kaggle link and place it in the same folder.
3. Run the Jupyter Notebook file to train and save the five models.
   - After execution, the trained models will be saved in your project directory.
4. Start the application by running the following command in your terminal:
   ```bash
   python app.py
   ```
5. Open the provided link in your browser.
6. Upload dermoscopic images and review predictions from all five models.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- Keras
- Flask
- NumPy
- Pandas
- Matplotlib

You can install them using:
```bash
pip install -r requirements.txt
```
## Sscreenshots
![Screenshot 2025-01-06 114208](https://github.com/user-attachments/assets/4e50c123-dd94-41cb-b549-d83566849529)

after predicting:
![Screenshot 2025-01-06 114233](https://github.com/user-attachments/assets/540f616d-0e6a-464d-95cc-e35b8dda251c)


## Future Improvements
- Increase dataset size for better accuracy.
- Optimize model architectures.
- Implement real-time image processing.

## Acknowledgments
Special thanks to ISI - International Skin Imaging Centre for providing the dataset.

## License
This project is licensed under the MIT License.

---
For any issues or contributions, feel free to open an issue or submit a pull request.

