import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import dtreeviz

# Load and prepare the df:
file_path = './data/survey_lung_cancer_data.csv'
df = pd.read_csv(file_path)
df .head()
print('head: ', df.head())
# Some analysis on the numerical columns
df.describe()
print('describe: ', df.describe())

X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
y = df['LUNG_CANCER']

# Split the dataset into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Create and train the decision tree classifier:
clf = DecisionTreeClassifier(criterion='gini', max_depth=4,
                             min_samples_leaf=3, random_state=1)

# clf = DecisionTreeClassifier(criterion='entropy', max_depth=4,
#                              min_samples_leaf=3, random_state=1)
# clf = DecisionTreeClassifier(criterion='log_loss', max_depth=4,
#                              min_samples_leaf=3, random_state=1)

# Train Decision Tree classifier
clf = clf.fit(X_train, y_train)
print(f"Training accuracy: {clf.score(X_train, y_train)}")
print(f"Testing accuracy: {clf.score(X_test, y_test)}")

# Make predictions:
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("score:", clf.score(X_test, y_test))

print("classification_report: ")
print(metrics.classification_report(y_test, y_pred))

allergy = df.get('FATIGUE')
df['LUNG_CANCER'].nunique()
df['LUNG_CANCER'].unique()
null_columns = df.isnull().any()
features = list(df .columns[:-1])

# Plot the decision tree
plt.figure(figsize=(15, 7))
plot_tree(clf, filled=True, feature_names=X.columns,
          class_names=True, rounded=True)
plt.show()

# dtreeviz
viz = dtreeviz.model(clf,
                     X_train,
                     y_train,
                     target_name='LUNG_CANCER',
                     feature_names=X.columns.tolist(),
                     class_names=[0, 1])
v = viz.view()
v.show()
v.save("./result/lung_cancer_1.svg")
