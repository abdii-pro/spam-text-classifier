import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

data = {
    'Text': [
        'Congratulations! You have won a lottery',
        'Reminder: your appointment is tomorrow at 10AM',
        'Win money now!!! Click here',
        'Meeting scheduled with your manager at 3PM', 'Limited time offer! Claim your free prize',
        'Please review the attached report', 'You have been selected for a cash prize', 'Can you send me the presentation?',
        'URGENT: Your account has been compromised!', 'Lets catch up for lunch next week'],
    'Label': [
        'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
x_encoded = vectorizer.fit_transform(df['Text'])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Label'])

x_train, x_test, y_train, y_test = train_test_split(x_encoded,y_encoded,test_size=0.3,random_state=42,stratify=y_encoded)


model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuraccy: ",accuracy_score(y_test,y_pred))
print('Actual Label Distributation',np.bincount(y_test))
print('Predicted Label Distribution',np.bincount(y_pred))
print('Classification Report',classification_report(y_test,y_pred,target_names=label_encoder.classes_,zero_division=0))

new_message = ['Win a free iykjkktjtgu trtrt rttr phone by clicking this link']
new_vector = vectorizer.transform(new_message)
predicted_class = model.predict(new_vector)
predicted_label = label_encoder.inverse_transform(predicted_class)[0]

print('Prediction for a new Class',predicted_label)