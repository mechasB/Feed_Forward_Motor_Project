# 🏎️ FSP 25 - Power Prediction & Analysis

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![Event](https://img.shields.io/badge/event-FSP_25-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 About the Project

This repository houses the source code and research documentation regarding the **analysis and prediction of power consumption** for a Formula Student race car. The project is executed as part of the preparations for **Formula Student Poland 2025 (FSP 25)**.

We utilize **real telemetry data** collected from previous track runs and testing sessions to develop Machine Learning (ML) models designed to:
1. Predict instantaneous power demand based on dynamic conditions.
2. Optimize **Energy Management** strategies for the Endurance event.
3. Analyze the correlation between driver inputs and energy consumption.

## 👥 Authors

The project is developed by the Data Science / Electronics sub-team:

| Name | Role 
| :--- | :--- 
| **Michał Błotniak** | **Project Lead**<br>Embedded Developer @ PUT Motorsport|
| **Mateusz Gogola** | **Data Scientist** | x |


## 🎯 Research Goals

The primary objective of this project is to develop a machine learning model that achieves the **highest possible prediction accuracy** while maintaining **real-world feasibility** for deployment.

We aim to solve the optimization problem of maximizing model performance while minimizing computational complexity, ensuring the solution can be potentially implemented on the vehicle's embedded hardware (ECU/VCU or Onboard Computer).

Key objectives include:

1.  **High-Fidelity Prediction:** Minimizing the error between predicted and actual power consumption ($P_{pred} \approx P_{real}$) to allow for precise Energy Management strategies.
2.  **Realizability & Constraints:** Ensuring the model architecture is lightweight enough to run with low latency. We prioritize models that are:
    * **Computationally Efficient:** Low inference time suitable for real-time or near real-time applications.
    * **Memory Efficient:** Compatible with limited resources of embedded systems.
3.  **Robustness:** The model must generalize well to unseen data (e.g., different tracks or changing weather conditions) without overfitting.

## 💾 Data Structure

Input data is retrieved from the vehicle's Data Acquisition System (DAQ) and includes:
* **HV Battery:** Voltage, Current, Cell Temperatures.
* **Inverters:** Motor RPM, Commanded Torque vs. Actual Torque.

> **Note:** Due to the confidentiality of the vehicle's technical specifications, raw `.csv`/`.log` files are not publicly available in this repository.

## 🛠️ Requirements & Installation

The project requires Python 3.8+. To set up the environment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-ORG/fsp25-power-models.git](https://github.com/YOUR-ORG/fsp25-power-models.git)
    cd fsp25-power-models
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Jupyter Notebooks:**
    ```bash
    jupyter notebook notebooks/
    ```

## 📂 Repository Structure

```text
├── data/               # Mock data for testing
├── src/                # Source code (models, data pipelines)
├── reports/            # Generated plots and performance reports
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
