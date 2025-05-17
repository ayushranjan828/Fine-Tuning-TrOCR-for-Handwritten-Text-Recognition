# 📝 Fine-Tuning TrOCR for Handwritten Text Recognition

This project focuses on fine-tuning [Microsoft's TrOCR model](https://huggingface.co/microsoft/trocr-base-handwritten), a Transformer-based Optical Character Recognition system, for improved performance on handwritten English text using custom datasets.

## 📌 Overview

We fine-tuned the `trocr-base-handwritten` model using a combination of the IAM Handwriting Database and the Imgur5K dataset. The goal was to build a robust handwriting recognition system capable of generalizing across diverse writer styles and noisy image conditions.

## 🧠 Model & Datasets

- **Model**: [TrOCR Base Handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)
- **Primary Dataset**: Custom subset of IAM Handwriting Database (2023)
- **Supplementary Dataset**: Imgur5K — real-world handwritten words
- **Format**: CSV file (`english.csv`) with `image` and `label` columns

## ⚙️ Preprocessing & Training

### Image Preprocessing

- Resized all images to `384x384`
- Converted to grayscale (3-channel RGB)
- Normalized using PyTorch `transforms.ToTensor()`

### Text Preprocessing

- Tokenized using `TrOCRProcessor`
- Cleaned Unicode and spacing issues
- Used `max_length=128`, truncation, and padding

### Training Configuration

- **Optimizer**: AdamW (`lr=5e-5`)
- **Epochs**: 10
- **Batch Size**: 4 (optimized for 16GB GPU using AMP)
- **Loss Function**: Cross-entropy (`VisionEncoderDecoderModel`)
- **Mixed Precision**: Enabled with `torch.cuda.amp`

## 📈 Evaluation Metrics

| Metric                         | Score   |
|--------------------------------|---------|
| **Character Error Rate (CER)** | 6.32%   |
| **Word Error Rate (WER)**      | 13.77%  |

✅ Achieves target thresholds (CER ≤ 7%, WER ≤ 15%)

## 🧩 Challenges Faced

- GPU memory limitations (mitigated with mixed precision)
- Variability in handwriting styles and label noise
- Sequence length imbalance across dataset

## 🚀 Future Enhancements

- Use [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) for synthetic data augmentation
- Apply curriculum learning: train on clean data, then gradually add noisy samples
- Experiment with larger models like `trocr-large`
- Improve decoding with beam search

## 📌 Conclusion

The fine-tuned TrOCR model delivers strong performance on handwritten OCR tasks and meets the evaluation criteria. It is ready for real-world applications such as document digitization and archival systems.

---

Feel free to fork this project or open issues for collaboration!

