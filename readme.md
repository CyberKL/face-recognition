# Facial Recognition Biometric System

This project implements two facial recognition methods:

- **Technology Evaluation:**  
  - Eigenfaces (PCA-based)  
  - FaceNet embeddings with MTCNN face detection  
  - Evaluated on LFW dataset with train/test split

- **Scenario Evaluation:**  
  - Real-time webcam-based access control app  
  - User registration with GUI username input  
  - Recognition on demand with sound feedback and logging

---

## Setup

1. Clone the repo and navigate into it.

2. Install dependencies:

```bash
pip install --no-deps facenet-pytorch
pip install -r requirements.txt
