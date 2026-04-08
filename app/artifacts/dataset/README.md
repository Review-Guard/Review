# Downloaded Datasets

This folder contains datasets downloaded from public internet sources for fake review detection experiments.

## 1) Fake Reviews Dataset (CSV)
- File: `fake_reviews_dataset_sayamalt.csv`
- Source: `https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv`
- Size: ~15.3 MB
- Notes:
  - Columns include: `category`, `rating`, `label`, `text_`
  - Label values include entries like `CG` and `OR` (use your preprocessing script to map these to binary classes).

## 2) Deceptive Opinion Spam Corpus v1.4
- Archive: `op_spam_v1.4.zip`
- Extracted folder: `op_spam_v1.4/op_spam_v1.4/`
- Source: `https://myleott.com/op_spam_v1.4.zip`
- Notes:
  - Contains 1600 text reviews organized by polarity and truthfulness/deception folders.
  - Labels are inferred from folder structure (for example, `deceptive_from_MTurk` vs `truthful_from_*`).

## Additional source inspected
- Folder: `FakeReview_Dataset_repo/`
- Source repo: `https://github.com/FakeReview/Dataset`
- Note: This repository provides documentation and contact details, but does not include directly downloadable dataset files in the repo itself.
