# task1
CNN-models-on-Student performance prediction
Project Overview
This project demonstrates how CNNs can be used to predict student performance based on multiple behavioural, social and environmental factors. 
This model uses these factors to predict a normalized score representing a student's exam performance. The results highlight the potential for educational institutions to gain insights into factors that affect performance and improve student outcomes.
Two models were developed and trained for this purpose:
Both models use:
1.	Conv2D layers with 32 and 64 filters.
2.	MaxPooling2D layers after each convolution to reduce the dimensionality.
3.	Flatten layer to convert the 2D data into a 1D vector.
4.	Dense layer with 64 neurons using ReLU activation.
5.	Final Dense layer using softmax for multiclass classification
Despite using the same architecture, Model 1 and Model 2 may converge differently or have subtle performance differences because of the randomness introduced by the optimizer (Adam). 
Even though the models are the same, the weights of CNN layers are randomly initialized when training starts. This randomness can lead to different results in accuracy, precision, and F1 scores.
The core difference between the two models lies not in their architecture but rather in the training process, which involves random initialization and optimization steps, resulting in slight variations in their final performance (accuracy, precision, F1 score). 
Problem Statement
	Student performance is influenced by multiple factors, many of which are non-linear and interdependent. Traditional statistical models fail to capture the complex relationships among these factors. 
	In this project, through deep learning techniques we are tasked with predicting student exam scores based on a variety of factors using a multiclass classification approach. The goal is to identify how different behavioural, environmental, and personal factors impact the likelihood of a student achieving a specific score range.
	The dataset used contains various features that influence a student's performance, allowing us to model this complex relationship. 
	The target variable in this dataset is Exam_Score, which represents the student's actual performance in exams. The dataset presents a typical multiclass classification problem, as the exam scores are grouped into multiple categories (e.g., score ranges like low, medium, high).
Dataset Features
The dataset includes the following key features, which can broadly be classified into:
1.	Academic and Study Habits:
Hours_Studied: The number of hours a student spends studying.
Previous_Scores: Scores from previous tests or assessments.
Attendance: How regularly the student attends classes.
Tutoring_Sessions: Additional academic help outside of regular classes.
2.	Personal and Behavioral Factors:
Sleep_Hours: The number of hours a student sleeps daily.
Motivation_Level: A subjective measure of the student’s motivation to perform well.
Physical_Activity: Engagement in physical exercises, which may influence mental alertness and focus.
3.	Environmental and Parental Support:
Parental_Involvement: Measures how involved parents are in the student’s education.
Parental_Education_Level: Educational background of the student’s parents.
Family_Income: Socioeconomic status based on family income.
Teacher_Quality: A measure of how students perceive the quality of teaching.
4.	School and Peer Environment:
School_Type: Whether the school is public or private.
Peer_Influence: The degree to which peers influence the student’s behavior and academic performance.
Access_to_Resources: Availability of educational materials, internet, etc.
Results
After training and evaluation, the performance of both models was measured using accuracy, precision, and F1-score:
• Model 1:
•	Accuracy: 32.45 %
•	Precision: 14.39%
•	F1 Score: 12.15 %
• Model 2:
•	Accuracy: 29.12%
•	Precision: 10.45 %
•	F1 Score:  8.78%


Process of implementation the Code
1.	Preprocessing:
The data is normalized using StandardScaler.
Categorical features are one-hot encoded.

2.	Model Training:
Two CNN models are trained for 90 epochs with a batch size of 32.
categorical_crossentropy is used as the loss function.
Adam optimizer is used.

3.	Evaluation Metrics:
The performance is evaluated using accuracy, precision (macro), and F1 score (macro).
