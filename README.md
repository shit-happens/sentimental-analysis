# Sentiment Analysis for Customer Queries  

**Project Overview**  
A retail firm named **Poseidon** has integrated a chatbot into its progressive web application to interact with customers. The goal of this project was to determine the core intent behind customer interactions (queries) and classify them into one of the following 10 predefined tags:  

| Tag ID | Intent Description              |  
|--------|---------------------------------|  
| 0      | Order Cancellation              |  
| 1      | Order Concern (Delay)           |  
| 2      | Order Modification              |  
| 3      | Order Status Inquiry            |  
| 4      | Product Browsing                |  
| 5      | Product Reviews                 |  
| 6      | Shipping Address Modification   |  
| 7      | Shipping Plans Browsing         |  
| 8      | Store Browsing                  |  
| 9      | Store Timings Inquiry           |  

**Objective**  
The primary objective was to classify customer queries into the above categories using a deep learning model, enabling Poseidon to respond effectively to customer needs.

---

## **Approach**  

### **Data Processing**  
Customer queries were treated as text data, and the tags served as the target classification labels. The processed text data was used to train a machine learning model.  

### **Model Architecture**  
An **LSTM (Long Short-Term Memory)** network, a type of Recurrent Neural Network (RNN), was employed for this task due to its ability to process sequential data effectively. LSTM networks are particularly well-suited for text analysis as they:  
- Retain important information from earlier steps in the sequence.  
- Handle long-term dependencies better than standard RNNs.  

---

## **Model Training and Evaluation**  

### **Training Process**  
The model was trained on the labeled dataset, and its performance was monitored using training and validation loss curves.  

### **Results**  
- **Validation Accuracy**: The model achieved a solid **83.98% accuracy** on the validation set.  
- **Insights**: The training and validation loss plots indicated effective learning with minimal overfitting.  

---

## **Conclusion**  
The LSTM model successfully classified customer queries into intent tags, demonstrating its potential to enhance Poseidon’s chatbot accuracy and customer satisfaction. Future improvements could include:  
- Incorporating attention mechanisms for better context understanding.  
- Expanding the dataset to include more diverse interactions.  
