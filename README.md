# Vehicle Wheel Torque Model
This project focuses on creating a wheel torque model for a vehicle. Using real-world data, the wheel torque is estimated in MATLAB based on parameters such as motor torque and wheel speed. The project uses deep learning techniques to learn and create a wheel torque model.

## Features
- Estimates wheel torque using real-world data.
- Utilizes motor torque, wheel speed, and other relevant parameters.
- Applies deep learning to model wheel torque.
- MATLAB-based implementation.
- Jupyter Notebook (`main.ipynb`) for running the workflow.

## Requirements
- MATLAB
- CARSIM
- Python

## How to Use
1. Open JupyterLab.
2. Open `main.ipynb` and set the necessary paths for your MATLAB model and data files.
3. Run all cells in the notebook to execute the vehicle wheel torque estimation workflow.

## Workflow
1. **Data Preprocessing**:  
   Load and preprocess real-world data, including motor torque, wheel speed, and other relevant parameters.
2. **Deep Learning Model**:  
   Implement the deep learning model in MATLAB to estimate wheel torque based on the input parameters.
3. **Model Training**:  
   Train the deep learning model using the preprocessed data.
4. **Model Evaluation**:  
   Evaluate the model’s performance by comparing the estimated wheel torque with actual measured data.
5. **Analysis**:  
   Analyze the results and refine the model as needed.

## Results
The tool outputs:
- A trained deep learning model capable of estimating wheel torque.
- Performance metrics showing the accuracy of the model’s predictions.
