import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


chun = pd.read_csv(
    r'/Users/apple/Documents/traffic_engineering/kaggle_practice/Churn_Modelling.csv')

print(chun.shape)
'''
print(chun.head)
# create the default size for graphs
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size

# draw the pie plot for the Exited column
chun.Exited.value_counts().plot(kind='pie', autopct='1.0f%%', colors=[
    'skyblue', 'orange'], explode=(0.05, 0.05))
plt.show()
# The output shows that in our dataset, 20% of the customers left the bank. Here 1 belongs to the case where the customer left the bank, where 0 refers to the scenario where a customer didn't leave the bank.


# plot the number od customers from all geographical locations in the dataset
sns.countplot(x='Geography', data=chun)
# The output shows that almost half of the customers belong to France, while the ratio of customers belonging to Spain and Germany is 25% each.

# plot number of customers from each unique geographical location along with the customer churn information
sns.countplot('x=Exited', hue='Geography', data=chun)

#The output shows that though the overall number of French customers is twice that of the number of Spanish and German customers, the ratio of customers who left the bank is the same for French and German customers. Similarly, the overall number of German and Spanish customers is the same, but the number of German customers who left the bank is twice that of the Spanish customers, which shows that German customers are more likely to leave the bank after 6 months.

plt.show()
'''
# There are a few columns that can be treated as numeric as well as categorical. For instance, the HasCrCard column can have 1 or 0 as its values. However, the HasCrCard columns contains information about whether or not a customer has credit card. It is advised that the column that can be treated as both categorical and numerical, are treated as categorical. However, it totally depends upon the domain knowledge of the dataset.
# Let's again print all the columns in our dataset and find out which of the columns can be treated as numerical and which columns should be treated as categorical. The columns attribute of a dataframe prints all the column names:

print(chun.columns)

# From the columns in our dataset, we will not use the RowNumber, CustomerId, and Surname columns since the values for these columns are totally random and have no relation with the output.
# Among the rest of the columns, Geography, Gender, HasCrCard, and IsActiveMember columns can be treated as categorical columns
# create a list of useful and related columns(seperated by catogory and numerical)

categorical_columns = ['Geography', 'Gender', 'HasCrCard',
                       'IsActiveMember']
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance',
                     'NumOfProducts', 'EstimatedSalary']

# finally, the output(the value from Exited columns are stored in the outputs variable)
outputs = ['Exited']
# check the datatype of all columns, catogory is a datatype
print(chun.dtypes)

for category in categorical_columns:
    chun[category] = chun[category].astype('category')
    print(chun.dtypes)

# see all caregory in Geography!(you need to remember)
print(chun['Geography'].cat.categories)

# When you change a column's data type to category, each category in the column is assigned a unique code. For instance, let's plot the first five rows of the Geography column and print the code values for the first five rows:
print(chun['Geography'].head())

# plot the codes for the values in the first five rows of Geography numerical_columns

print(chun['Geography'].head().cat.codes)
# The output shows that France has been coded as 0, and Spain has been coded as 2.

# (the numerical converting probelm has been solved)The basic purpose of separating categorical columns from the numerical columns is that values in the numerical column can be directly fed into neural networks. However, the values for the categorical columns first have to be converted into numeric types. The coding of the values in the categorical column partially solves the task of numerical conversion of the categorical columns.


# Since we will be using PyTorch for model training, we need to convert our categorical and numerical columns to tensors.

# first convert the categorical columns to tensors. In PyTorch, tensors can be created via the numpy arrays. We will first convert data in the four categorical columns into numpy arrays and then stack all the columns horizontally, as shown in the following script:
geo = chun['Geography'].cat.codes.values
gen = chun['Gender'].cat.codes.values
hcc = chun['HasCrCard'].cat.codes.values
iam = chun['IsActiveMember'].cat.codes.values

categorical_data = np.stack([geo, gen, hcc, iam], 1)

print(categorical_data[:10])

# Now to create a tensor from the aforementioned numpy array, you can simply pass the array to the tensor class of the torch module. Remember, for the categorical columns the data type should be torch.int64.
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
print(categorical_data[:10])

# In the output, you can see that the numpy array of categorical data has now been converted into a tensor object.
# In the same way, we can convert our numerical columns to tensors:

numerical_data = np.stack([chun[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)
print(numerical_data)

# In the output, you can see the first five rows containing the values for the six numerical columns in our dataset.

# The final step is to convert the output numpy array into a tensor object.

outputs = torch.tensor(chun[outputs].values).flatten()
print(outputs[:5])


# Let now plot the shape of our categorial data, numerical data, and the corresponding output:
print(categorical_data.shape)
print(numerical_data.shape)
print(outputs.shape)


# However, a better way is to represent values in a categorical column is in the form of an N-dimensional vector, instead of a single integer. A vector is capable of capturing more information and can find relationships between different categorical values in a more appropriate way. Therefore, we will represent values in the categorical columns in the form of N-dimensional vectors. This process is called embedding.

# We need to define the embedding size (vector dimensions) for all the categorical columns. There is no hard and fast rule regarding the number of dimensions. A good rule of thumb to define the embedding size for a column is to divide the number of unique values in the column by 2 (but not exceeding 50). For instance, for the Geography column, the number of unique values is 3. The corresponding embedding size for the Geography column will be 3/2 = 1.5 = 2 (round off).
# unique number / 2  --->> round off then 3/2 = 1.5 = 2


# The following script creates a tuple that contains the number of unique values and the dimension sizes for all the categorical columns:
categorical_columns_sizes = [len(chun[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2))
                               for col_size in categorical_columns_sizes]
print(categorical_embedding_sizes)

# A supervised deep learning model, such as the one we are developing in this article, is trained using training data and the model performance is evaluated on the test dataset. Therefore, we need to divide our dataset into training and test sets as shown in the following script:

total_records = 10000
test_records = int(total_records * .2)

total_records = 10000
test_records = int(total_records * .2)

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
numerical_train_data = numerical_data[:total_records-test_records]
numerical_test_data = numerical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]
# We have 10 thousand records in our dataset, of which 80% records, i.e. 8000 records, will be used to train the model while the remaining 20% records will be used to evaluate the performance of our model. Notice, in the script above, the categorical and numerical data, as well as the outputs have been divided into the training and test sets.
# To verify that we have correctly divided data into training and test sets, let's print the lengths of the training and test records:
print(len(categorical_train_data))
print(len(numerical_train_data))
print(len(train_outputs))

print(len(categorical_test_data))
print(len(numerical_test_data))
print(len(test_outputs))
