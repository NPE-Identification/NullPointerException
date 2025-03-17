# Replication package for paper "LLM-Based Identification of Null Pointer Exception Patches"

![AACC Proposed Approach](AACC%20Proposed%20Approach.png)

LLM-Based Identification of Null Pointer Exception Patches This repo uses Large Language Models (LLMs) to identify and classify Null Pointer Exception (NPE) patches in code commits, combining contextual embeddings and AST features for automated NPE fix Identification and classification in software development.

# Augmented Agentic Commit Classification (AACC)  
## Identifying and Classifying Null Pointer Exception (NPE) Fixes  

## 🚀 Overview  

Null Pointer Exceptions (**NPEs**) are one of the primary causes of **software crashes** and **runtime errors**, making them a significant challenge for developers.  

Existing approaches for identifying and classifying NPE-related fixes often suffer from:  
- **Noisy or irrelevant data**  
- **Lack of contextual understanding**  
- **Inefficient processing of large and imbalanced datasets**  

To overcome these limitations, we introduce **Augmented Agentic Commit Classification (AACC)** — a novel method designed to accurately classify commit patches as **NPE fixes or non-NPE**.  

📊 **Performance Boost**: Our evaluation using **ChatGPT-4o** shows that AACC **outperforms state-of-the-art models**, improving the **F1 score from 72.07% to 98.03%**.

---

## 🔑 Key Features  

AACC incorporates four major advancements to enhance the detection of NPE fixes:  

### ✅ Best Example Selection  
- Filters **high-quality, contextually relevant commits**, ensuring the model learns from meaningful and accurate data.  

### 📚 Augmented Knowledge Base  
- Combines **contextual metadata, program semantics, and bug-fix patterns** for enriched classification.  

### 🎯 Prioritize Agent  
- **Ranks commits based on relevance and impact**, optimizing resource allocation and improving classification efficiency.  

### 🔄 Iterative Refinement Process  
- Uses **feedback loops** to **learn from misclassifications**, reducing false negatives and improving overall accuracy.  

---

## 📁 Folder Structure  
📦 AACC/ │── 📂 Dataset/ # Labeled datasets for training, testing and evaluation │ ├── 📝 NPEPatches.json # commit patches and messages, stack trace and meta data of commits │ ├── 📝 preprocessed.json # Cleaned and structured dataset │ ├── 📝 labeled_data # Human-annotated NPE-fixes and non-NPE commits │ │── 📂 Baselines_Evaluation/ # Comparative evaluation of AACC vs. existing approaches │ ├── 📜 models/ # Implementations of baseline models │ ├── 📜 performance_metrics/ # F1 scores, accuracy, and comparative results │ ├── 📜 evaluation_scripts/ # Python scripts for evaluation │ │── 📂 LLMS/ # LLM-based classification models and components │ ├── 🤖 fine_tuned_models/ # Pre-trained models adapted for NPE classification │ ├── 📝 prompt_engineering/ # Strategies and examples for LLM-based classification │ ├── 📜 inference_scripts/ # Scripts to classify new commit patches │ │── 📜 run_classification.py # Main script to run AACC classification │── 📜 requirements.txt # Required dependencies │── 📜 README.md # Project documentation

---
## ⚙️ Getting Started  
## 📌 Requirements  

To run the **Augmented Agentic Commit Classification (AACC)** project, you need to have the following dependencies installed. These dependencies can be easily installed using `pip`.  

### 🖥️ System Requirements  
- **Python 3.8 or higher**  

### 📦 Python Libraries  
- **torch**: Required for PyTorch-based model embeddings and deep learning operations.  
- **transformers**: Provides pre-trained language models for commit classification.  
- **sentence-transformers**: Used for generating contextual embeddings of commit messages.  
- **pandas**: Used for reading, processing, and handling structured datasets.  
- **scikit-learn**: Provides machine learning utilities for classification and evaluation.  
- **openai**: Required to interact with OpenAI models like **ChatGPT-4o** for enhanced commit analysis.  
- **anthropic**: Used to interact with **Claude Sonnet** for advanced language processing.  
- **groq**: Required for using **DeepSeek-r1** and **Llama** models via Groq API.  

### 🌐 API Keys  
This project integrates multiple **LLMs** for commit classification. Ensure you have API keys set up for:  

- **Groq API** (`GROQ_API_KEY`) → Supports **DeepSeek-r1** and **Llama** models.  
- **OpenAI API** (`OPENAI_API_KEY`) → Used for **ChatGPT-4o** and other OpenAI models.  
- **Gemini API** (`GEMINI_API_KEY`) → Required for Google’s **Gemini AI**.  
- **Anthropic API** (`ANTHROPIC_API_KEY`) → Used for **Claude Sonnet**.  

Store these keys in your environment variables or in a `.env` file for secure access.

```bash
export GROQ_API_KEY="your-groq-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"
export GEMINI_API_KEY="your-gemini-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"






## Project Overview
The objective of this project is to provide a novel approach to detecting data contamination in deep learning models. By leveraging the **naturalness** of source code and combining it with the model’s performance, this approach significantly outperforms traditional methods for detecting whether a piece of code has been used for model training.
## Features
- Detection of contaminated vs. cleaned datasets using naturalness and model performance.
- Works for code completion models and method name suggestion tasks.
- Code and dataset collection for constructing contaminated (𝐶𝑇𝑑𝑎𝑡𝑎) and cleaned (𝐶𝐿𝑑𝑎𝑡𝑎) datasets.

## Directory Structure

This section provides an overview of the repository structure, including datasets, and the source code of the Code Completion models: [Dataset](https://drive.google.com/file/d/1QheSAfupFNCq_V4q4a4Mt8uHNDl_gpC2/view?usp=sharing),[UniXcoder](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/UniXcoder), [CodeParrot](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/CodeParrot), and [Ngram Model](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/n-gram_cachelm).
![Overview](directories.PNG)

# Dataset for Code Completion and N-gram Models
<p align="center">
    <img src="DatasetGraph.PNG" alt="TestingDataset" width="600">
</p>



This folder contains all the datasets used for training and testing the models in the paper **"Has My Code Been Stolen for Model Training? A Naturalness-Based Approach to Code Contamination Detection"**. The dataset is organized into three main parts: **Train Dataset**, **Testing Dataset for Code Completion Models**, and **N-gram Train Dataset**.

## Folder Structure
```plaintext
dataset/
│
├── Train Dataset (for Code Completion Model)
│   ├── UniXcoder Training Data
│   │   ├── file1.java
│   │   ├── file2.java
│   │   └── ...
│   └── CodeParrot Training Data
│       ├── file1.java
│       ├── file2.java
│       └── ...
│
├── N-gram Train Dataset (for N-gram Model)
│   ├── file1.java
│   ├── file2.java
│   └── ...
│
└── Test Dataset (for N-gram Model)
    ├── ChatGPT3.5, Claude, UniXcoder and CodeParrot Data
    │   ├── CLdata
    │   │   ├── file1.java
    │   │   ├── file2.java
    │   │   └── ...
    │   └── CTdata
    │       ├── file1.java
    │       ├── file2.java
    │       └── ...
└── Test Dataset (for Code Completion Models)
    │   ├── CLdata
    │   │   ├── file1.java
    │   │   ├── file2.java
    │   │   └── ...
    │   └── CTdata
    │       ├── file1.java
    │       ├── file2.java
    │       └── ...
```
## Download Dataset

Due to size limits, the dataset has been uploaded to Google Drive. You can download the dataset using the following link:

- [Download Dataset from Google Drive](https://drive.google.com/file/d/1QheSAfupFNCq_V4q4a4Mt8uHNDl_gpC2/view?usp=sharing)
  
### 1. Train Dataset (for Code Completion Model)
This folder contains the training datasets for two code-completion models: **UniXcoder** and **CodeParrot**.


- **`UniXcoder/CodeParrot Training Data`**: 
    - Contains Java files that are used to train the UniXcoder\CodeParrot model.
    - Example files: `file1.java`, `file2.java`, etc.
    - For details on training these models, refer to the respective repositories:
        - [UniXcoder](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/UniXcoder)
        - [CodeParrot](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/CodeParrot)
    
### 2. `Train Dataset (for N-gram Model)`
This folder contains the data used to train the n-gram model. It includes various Java files:

- Example files: `file1.java`, `file2.java`, etc.
To train the N-gram model, you can use the code provided in the [Ngram Model](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/n-gram_cachelm) repository. Follow the instructions in the repository to preprocess the data and train the model.


### 3. `Test Dataset (for Code Completion and Ngram Models)`

This folder contains the test datasets for **(UniXcoder** and **CodeParrot)** and pre-trained models **(ChatGPT3.5 and Claude)**. The test data is split into two categories:

- **`CLdata/`** (Cleaned Data):
    - It contains clean Java files not part of the code completion model training data.
    - Example files: `file1.java`, `file2.java`, etc.
  
- **`CTdata/`** (Contaminated Data):
    - Contains Java files that were part of the code completion model training data(as shown in dataset fig).
    - Example files: `file1.java`, `file2.java`, etc.


## Usage Instructions

- **Train Dataset**: Use the files in the `Train Dataset (for Code Completion Model)` folder for training the UniXcoder and CodeParrot models.
- **N-gram Train Dataset**: The `N-gram Train Dataset` folder contains the data required to train an n-gram model, which is used for code naturalness evaluation.
- **Test Dataset**: To evaluate, use the files in the `Test Dataset for Code Completion Models(UniXcoder, CodeParrot, ChatGPT3.5 and Claude) and Ngram Model`. The `CLdata` folder contains clean test data, while the `CTdata` folder contains contaminated test data.



## Steps for Contamination Detection

### 1. **Performance and Naturalness:**
  - **Performance:** Calculate the token-level accuracy from Code Completion Models(Unixcoder, Codeparrot, ChatGPT3.5, Claude).
  - **Naturalness:** Evaluate the naturalness scores for these snippets using the N-gram model.

### 2. **Train the Classifier**
-Code-Completion-Model
    - Combine performance metrics and naturalness scores.
    - Train an SVM classifier to distinguish between contaminated and cleaned data.
-Method-name prediction model
    - Combine Edit distance with the length of the method name and method body and the naturalness of the method name and method body.
    - Train an SVM classifier to distinguish between contaminated and cleaned data.
### 3. **Predict Contamination Source**
-Use the trained SVM classifier to predict whether a new code sample is contaminated (CTdata) or cleaned (CLdata).

### 4. **Usage**
To perform contamination detection for both the **Code Completion Model** and the **Method Name Prediction Model**, use the following scripts available in the repository:

- **For Code Completion Model:**

   To run the contamination detection for code completion, execute the `Classifier_performance.py` script:

   ```bash
   python Classifier_performance.py --datasets_dir path_to_your_datasets_folder
Replace path_to_your_datasets_folder with the actual path to your datasets folder containing performance and naturalness scores.

-- **For Method Name Prediction Model:**

To detect contamination using method name predictions, run the Classifier_methodname.py script:

   ```bash
   python Classifier_methodname.py --datasets_dir path_to_your_datasets_folder
```
Again, replace path_to_your_datasets_folder with the actual path to your dataset for method name prediction.

