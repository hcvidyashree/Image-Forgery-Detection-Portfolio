# Image Forgery Detection Using Image Processing

## Project Overview
This project explores image forgery detection using basic image processing techniques in Python. It compares original and potentially manipulated images to identify regions that may have been altered.

## Objectives
- Compare original and tampered images.
- Identify potential manipulated regions.
- Generate visual representations of detected differences.
- Analyze image similarity using image quality metrics.

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Scikit-image
- Pandas

## Detection Techniques
- Pixel difference analysis
- Edge inconsistency analysis
- Block similarity analysis
- Difference masks and heatmaps
- Bounding-box visualization

## Project Structure
```text
Image-Forgery-Detection/
├── detect.py
├── graph_analysis.py
├── requirements.txt
├── README.md
└── dataset/
    ├── graphs/
    └── output/
```

## Analysis Metrics
The project includes:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Absolute Mean Brightness Error (AMBE)
- Structural Similarity Index (SSIM)

## Important Note
The detection techniques and image quality metrics highlight differences between images. These differences alone do not establish whether an image has been forged. Results should be interpreted with care, particularly when image dimensions differ.

## Learning Outcomes
- Practical application of image processing.
- Image comparison and analysis.
- Visualization of potentially manipulated regions.
- Working with Python libraries for computer vision.

## Future Improvements
- Explore advanced image forgery detection techniques.
- Improve localization of manipulated regions.
- Evaluate performance on larger and more diverse datasets.

## Author
AI & Data Science Engineering Student

---

*An exploratory project focused on understanding image manipulation analysis through Python and computer vision.*
