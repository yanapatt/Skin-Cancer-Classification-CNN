# Skin-Cancer-Classification-CNN
---
## Project Description
โปรเจคนี้เกี่ยวกับการจำแนกภาพถ่ายโรคมะเร็งผิวหนัง (Classification) จำนวน 7 Classes ดังตารางด้านล่าง โดยใช้ความรู้ที่ได้เรียนในวิชา Computer Vision เพื่อพัฒนา Multimodal CNN with MLP model ในการหาความสัมพันธ์ระหว่าง Metadata ที่เป็น Tabuler Data ร่วมกับ Image data ที่เป็นข้อมูลภาพถ่าย เพื่อให้การทำนายในแต่ละ Class มีประสิทธิภาพมากยิ่งขึ้น

| Label | Total | Train | Augmented | Valid | Test |
|------|-------|-------|-----------|-------|------|
| bkl  | 1076  | 861   | 2605      | 117   | 98   |
| nv   | 6499  | 5211  | 5211      | 651   | 637  |
| df   | 115   | 92    | 368       | 12    | 11   |
| mel  | 1103  | 879   | 2605      | 107   | 117  |
| vasc | 142   | 116   | 464       | 15    | 11   |
| bcc  | 509   | 409   | 1636      | 52    | 48   |
| akiec| 327   | 264   | 1056      | 30    | 33   |

## Environment setup
All experiments were conducted on the Google Colab platform, utilizing an NVIDIA Tesla L4 GPU with VRAM 56.9 GB. for hardware acceleration. The models were implemented in Python using the TensorFlow
(v2.19.0) framework with its integrated Keras (v3.10.0) API. Other key libraries included Pandas for metadata manipulation and Matplotlib/Seaborn for the visualizations in our Exploratory Data Analysis (EDA).

## Dataset
HAM10000 Datasets: <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T">Click here</a>

## Model architecture
<img width="512" height="767" alt="Model" src="https://github.com/user-attachments/assets/778262ef-2220-402b-8bcc-583174738b19" />  

### Model Description
สถาปัตยกรรม Multimodal CNN with MLP ประกอบไปด้วย 2 Branch ได้แก่ Metadata Branch และ Image Branch  
* Metadata Branch: จะรับ 2 Inputs หลักได้แก่ Numerical Data และ Categorical Data จาก Tabular Data ที่ผ่านการ Normalization มาแล้ว สำหรับ Numerical Data ใช้ Standard Scaler และ Categorical ใช้ One-Hot Encoder และนำ 2 Inputs มา Concatenate กันเพื่อส่งให้ MLP (Multilayer Perceptron) ในการสกัด Feature จาก Metadata
* Image Branch: จะรับ 1 Input ได้แก่รูปภาพ ซึ่งผ่านการ Normalize ให้เหมาะสมต่อ Pretrained CNN Model ซึ่งจะใช้สำหรับการสกัด Feature จาก Image
* Attention: นำ Feature Vectors ที่ได้จากทั้ง 2 Branchs มา Concatenate กันและส่งไปยัง Dense Layer ที่ใช้ Sigmoid Activation และนำค่าจาก Feature Vector ก่อนหน้ามาคูณกัน (Multiply) เกิดเป็นค่า Weight ของทั้ง 2 Branchs และในขั้นตอนสุดท้ายจะนำค่า Weights เหล่านั้นมาบวกกัน (Addition) และส่งไปยัง Dense Layer สุดท้ายที่ใช้ Softmax Function ในการหาค่าความน่าจะเป็นของแต่ละ Class โดยจะใช้ Dropout Rate ที่ 0.5

## Result

All models
| Model                               | Accuracy | Precision | Recall | F1-Score |
|-------------------------------------|----------|-----------|--------|----------|
| EfficientNetV2B0 (Image Only)       | 83.46%   | 0.8534    | 0.8168 | 0.8272   |
| MobileNetV3Large (Image Only)       | 82.72%   | 0.8389    | 0.8178 | 0.8220   |
| XceptionNet (Image Only)            | 82.51%   | 0.8348    | 0.8094 | 0.8193   |
| **EfficientNetV2B0 (Multimodal)**   | **84.71%** | **0.8638** | **0.8304** | **0.8423** |
| MobileNetV3Large (Multimodal)       | 82.41%   | 0.8439    | 0.8094 | 0.8236   |
| XceptionNet (Multimodal)            | 82.30%   | 0.8301    | 0.8136 | 0.8181   |

EfficientNetV2B0 (Multimodal)
| Label        | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| akiec       | 0.5938    | 0.5758 | 0.5846   | 33      |
| bcc         | 0.6500    | 0.8125 | 0.7222   | 48      |
| bkl         | 0.7158    | 0.6939 | 0.7047   | 98      |
| df          | 0.7778    | 0.6364 | 0.7000   | 11      |
| mel         | 0.7381    | 0.5299 | 0.6169   | 117     |
| nv          | 0.9109    | 0.9466 | 0.9284   | 637     |
| vasc        | 0.8462    | 1.0000 | 0.9167   | 11      |
| **Accuracy**|           |        | **0.8471** | **955** |
| **Macro Avg** | 0.7475  | 0.7421 | 0.7391   | 955     |
| **Weighted Avg** | 0.8433 | 0.8471 | 0.8423 | 955     |


