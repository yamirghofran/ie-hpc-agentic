# %% [markdown]
# https://github.com/yamirghofran/ML-Fundamentals

# %% [markdown]
# # Problem: **Will a passenger survive?**

# %% [markdown]
# **Disclosure**: Generative AI (Claude Sonnet 3.5) was used for generating code for showing plots, more-than-normally-complex pandas operations, rare addition of comments for documentation, and formatted prints to display results in a well-formatted way (in tables, rounding up decimals, etc.). With the help of Generative AI, I did learn how to do them better on my own.

# %% [markdown]
# # Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
SEED = 42
from sklearn.neighbors import KNeighborsClassifier



# %%
# Show all columns
pd.set_option('display.max_columns', None)

# If you also want to see more rows
pd.set_option('display.max_rows', None)

# %% [markdown]
# # Data Loading and Initial Exploration

# %%
titanic_data_original = pd.read_excel('titanic3.xls')
titanic_data = titanic_data_original.copy()
titanic_data.head()

# %% [markdown]
# ### Dropping Leakage Columns (boat, body)

# %% [markdown]
# At first look, I notice that the `boat` attribute could cause leakage for multiple reasons:
# 1. Feature is from the future: During test time, when we use our ML model, we don't have the boat information for a person as an input. If we set the time of the target the specific moment of syncing, the boat attribute is obtained after that. Therefore it is from the future.
# 2. The boat attribute directly indicates the survival of the person. Therefore, the target can be considered a function of the boat attribute which would mislead our model since we don't have that information during test time.
# 
# For the same reasons, the `body` attribute could cause leakage because it is directly related to the survival of the passengers. If there is a body, then the passenger did not survive. And we don't have that information during test time.
# 
# As a result, it makes sense to drop these attributes from the beginning. No further processing is needed for these attributes.

# %%
titanic_data = titanic_data.drop(['boat', 'body'], axis=1)

# %% [markdown]
# ## Data Statistics and Visualizations
# Here we perform some preliminary analysis on the data with the pandas `describe()` function which outputs the common statistics of each attribute.
# 
# This doesn't necessarily give us powerful information, but allows us to have a decent image of the distribution of each attribute in our head for later analysis and maniputlation.

# %%
titanic_data.shape

# %%
titanic_data.describe()

# %% [markdown]
# Next, I will print the types of the columns to make sure what type of data I'm dealing with. This would influence how I would approach their imputation, encoding (if necessary, etc) based on if they are numerical vs categorical, discrete vs continuous, etc.

# %%
print(titanic_data.dtypes)


# %% [markdown]
# Futhermore, we use `seaborn` to plot a `pairplot` between the attributes to see if any relationship jumps out at us.

# %%
sns.pairplot(titanic_data)

# %% [markdown]
# There is no apparent correlation between the attributes that we can see from the pairplot. We will check this more rigorously when it comes to selecting the features.

# %% [markdown]
# ## Outliers

# %% [markdown]
# Next, we will create boxplots for the numerical features to see if there are any significant outliers. Outliers could potentially decrease the quality of our model because we don't have many records--only 1308--and we will use a shallow learning algorithm like Linear Regression which is quite sensetive to outliers and prone to overfitting because of them.

# %%
numerical_cols = ['age', 'fare', 'sibsp', 'parch']

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([titanic_data[col].dropna() for col in numerical_cols], labels=numerical_cols)

plt.title('Boxplots of Numerical Features')
plt.ylabel('Values')
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

# %% [markdown]
# From the boxplot, I can see that the outliers for `age`, `sibsp`, and `parch` aren't too dissimilar (far away from) the "maximum" which is defined as 3rd Quartile + 1.5 times the IQR. So, it's probably better to keep them because they might contain meaningful information, and keeping them won't compress the "meaningful" (non-outlier) data too much in the scaling stage. However, to be very rigorous, a log-transformation or winsorization could be used to minimize the effect of outliers and prevent overfitting.
# 
# The `fare` attribute, however, has many outliers that are far away from the "maximum". This could potentially hurt the quality of our model.
# 
# Removing outliers is not an obvious decision. Firstly, we would be removing data, which is not the best thing. Secondly, this would completely change the missing values we impute for the `fare` attribute, which could make our model different. However, in this case, it wouldn't be a big problem because there is only one missing value in the `fare` attribute.
# 
# I would like to test the model's performance in both cases of removing and not removing the outliers. All else being equal.
# 
# **Post-Testing Comment**: To test the impact of the outliers on the quality of the model, I decided to test different ways of dealing with the outiers in the `fare` attribute (after data split to avoid leakage and keep all real data for validation and test datasets):
# - Log Transformation: Compresses high values and preserves the relationship (distance) between points.
# - Winsorization: Set outliers to a percentile value (in this case 5th and 95th percentile)
# - Clipping: Set outliers to a standard deviation value (in this case 3 standard deviations)
# - Removing outliers: Justifiable if it leads to increased model performance, but not scientifically rigorous. Didn't help in this case.
# - Using `RobustScaler` instead of `StandardScaler` because it is more robust to outliers.
# 
# In all cases, the model's accuracy ended up being slightly lower or the same compared to ignoring (doing nothing about) outliers in the validation and test datasets.
# 
# This was a bit counter-intuitive at first because I suspected that outliers usually hurt the model's performance. Knowing the result of this experiment, however, I think a possible explanation is that  even though there are a lot of outliers in the `fare` attribute, they are the nature of what the pricing of the tickets actually were, not some noise or corruption in the data. Therefore, including them as they are was the best for the model's accuracy.

# %%
""" Code for handling outliers, applied after the train-test split, before feature selection.
# 1. Log transformation
X_train_balanced_log = X_train_balanced.copy()
X_train_balanced_log['fare'] = np.log1p(X_train_balanced['fare'])

# 2. Winsorization (capping at 5th and 95th percentiles)
X_train_balanced_winsor = X_train_balanced.copy()
lower = np.percentile(X_train_balanced['fare'], 5)
upper = np.percentile(X_train_balanced['fare'], 95)
X_train_balanced_winsor['fare'] = np.clip(X_train_balanced['fare'], lower, upper)

# 3. Clipping (capping at 3 std from mean)
X_train_balanced_clip = X_train_balanced.copy()
Q1 = X_train_balanced['fare'].quantile(0.25)
Q3 = X_train_balanced['fare'].quantile(0.75)
IQR = Q3 - Q1
X_train_balanced_clip['fare'] = np.clip(X_train_balanced['fare'],
                                       Q1 - 1.5*IQR,
                                       Q3 + 1.5*IQR)

# 4. Removal of outliers (removing points > 3 std from mean)
X_train_balanced_remove = X_train_balanced.copy()
outlier_mask = (X_train_balanced['fare'] >= Q1 - 1.5*IQR) & (X_train_balanced['fare'] <= Q3 + 1.5*IQR)
X_train_balanced_remove = X_train_balanced_remove[outlier_mask]
y_train_balanced_remove = y_train_balanced[outlier_mask]
"""

# %% [markdown]
# # Initial Feature Engineering
# 
# The `name` and `ticket` attributes don't have any missing values. They can't be used directly as features because they are categorical, yet have very high cardinality and aren't very informative on their own. 
# 
# Therefore, I will perform initial feature engineering steps before filling missing values.
# 

# %% [markdown]
# It's safe to do this "feature engineering" for these attributes before any split or other operation because the engineering relies solely on the value itself, and not anything else (other attributes or the target variable `survived`). It is only an issue of formatting and calculation of factual information. Therefore there is no risk of data leakage or data misshandling.

# %% [markdown]
# ## Name and Title Feature Engineering

# %% [markdown]
# In search for an improvement of my feature engineering, I got the idea, inspired by a [notebook online](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial), to extract the title from the `name` attribute. I would have to drop the `name` column anyway because it has extremely high . I suspect it will be correlated quite strongly with sex, but it could be a clever feature engineering technique. I will explore it more in the feature selection stage.

# %%
# Extract title from name
titanic_data['title'] = titanic_data['name'].str.extract(' ([A-Za-z]+)\.', expand=False) # Used AI to get the regex pattern to extract the title from the name.


# %%
plt.figure(figsize=(12, 6))
titanic_data['title'].value_counts().plot(kind='bar')
plt.title('Distribution of Titles')
plt.xlabel('Title')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %% [markdown]
# Looking at the distribution of titles, there are the common titles which have the most frequency: Mr, Miss, Mrs, Master. Then, there is a long tail of more rare titles. So for the sake of reducing the dimensions of this feature and avoiding the curse of dimensionality, I decided to "cut the long tail" and group all of them into a "Rare" title.

# %%
# Group rare titles into 'Rare'
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
titanic_data['title'] = titanic_data['title'].apply(lambda x: x if x in common_titles else 'Rare')


# %% [markdown]
# We can now safely drop the `name` attribute now that we have extracted the maximum information we could get from it.

# %%
titanic_data.drop('name', axis=1, inplace=True)
titanic_data.head()


# %% [markdown]
# ## Ticket Feature Engineering

# %% [markdown]
# Thinking about encoding categorical variables in the future and about the problem, I have to do something about the `ticket` attribute because it is not a numerical feature and there are too many unique values to encode with a method like one-hot encoding which would create a sparse high-dimension feature and introduce the curse of dimonsionality. This would be problematic because it would increase our computational complexity, increase the risk of overfitting, degrade the significance of distance metrics which could be used in model training, and reduce the interpretability of the model.
# 
# The only useful information that `ticket` gives us is if multiple people have the same ticket number. Therefore, I will synthesize a new feature `ticket_count` which will be the frequency of the ticket number in the dataset.
# 

# %%
titanic_data['ticket_count'] = titanic_data['ticket'].map(titanic_data['ticket'].value_counts()) #map the value counts of the ticket attribute to the ticket attribute.
titanic_data.head()

# %% [markdown]
# # Managing Missing Values (Before Split)

# %% [markdown]
# First, I want to get a good idea of how many missing values there are in the dataset and which attributes have the most missing values. I define a helper function to print the number and percentage of missing values for each attribute.

# %%
def display_null_values(df): # Defining a function to get the number and percentage of null entries in our dataset
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / len(df)) * 100
    
    null_info = pd.DataFrame({
        'Null Count': null_counts,
        'Null %': null_percentages.round(2)
    })
    
    return null_info.sort_values('Null Count', ascending=False)

print("\nNull Values Analysis:")
print(display_null_values(titanic_data))

# %% [markdown]
# There are a lot of missing values in the `cabin`, `home.dest`, and `age` attributes. Only 2 and 1 values are missing in the `embarked` and `fare` attributes respectively. We have to figure out how to handle these missing values for each attribute.

# %% [markdown]
# I want to impute `home.dest` before the data split because, as a part of the imputation, I want to look for shared ticket numbers for which there are some missing values but not all. This way, I can farely confidently and accurately impute `home.dest` I don't know how important this will be for the model, but I think it's a clever way. Anything else I will replace with "Unknown". There is no leakage that will happen because of this.

# %% [markdown]
# ## Missing values in `home.dest`
# 43% of the values in this attribute are missing and there isn't a clear pattern in the values. We can't use the mean or median to impute the missing values because it is a categorical variable. If we were to replace all missing values with the mode--most frequent value--we would probably cause a lot of overfitting since the model would learn a false pattern between survival and `home.dest`.
# 
# The standard method to impute a categorical sparse attribute like this is to just impute with "Unknown" because usually we can't confidently infer them from other values.
# 
# However, in this specific case, we can use the the `ticket` number to impute some of the missing values in `home.dest`. For each person with missing `home.dest`, we can look to find another person with a `home.dest` value with the same `ticket` number and set the `home.dest` value of the latter to the former.
# 
# This won't work for every missing value, however, so for the remaining missing values we will have to resort to replacing them with "Unknown".

# %%
# Create mapping of ticket to home.dest for non-null values
ticket_to_dest = titanic_data[titanic_data['home.dest'].notna()].groupby('ticket')['home.dest'].first()

# Fill missing home.dest values where possible using ticket matches
titanic_data.loc[titanic_data['home.dest'].isnull(), 'home.dest'] = (
    titanic_data.loc[titanic_data['home.dest'].isnull(), 'ticket']
    .map(ticket_to_dest)
)

# Fill remaining nulls with 'Unknown'
titanic_data['home.dest'] = titanic_data['home.dest'].fillna('Unknown')
titanic_data.drop(columns=['ticket'], inplace=True) #Drop ticket because we don't need it anymore.

# %% [markdown]
# Now, I will plot the values of `home.dest` to see the distribution of the values for future operations.

# %%
# Plot value counts of home.dest
plt.figure(figsize=(15, 6))
titanic_data['home.dest'].value_counts().plot(kind='bar')
plt.title('Distribution of Home Destinations')
plt.xlabel('Home Destination')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# %%
home_dest_counts = titanic_data['home.dest'].value_counts()
home_dest_counts


# %% [markdown]
# We can see that there is a very "long tail" of values which we will have to consider when performing the endoding of `home.dest`.

# %% [markdown]
# # Data Splitting
# I chose to split the data into training, validation, and test datasets at this stage because I plan to use statistical manipulations of data and prediction techniques for imputation from here on. Therefore, to not do that with the entire dataset and only training, in an effort to avoid data leakage, I will split the data here.
# 
# For splitting the data, I follow a standard practice:
# 1. Split the data randomly into train (70%) and temporary holdout (30%).
# 2. Solit the holdout randomly into validation and test (15% each).
# 
# I use the stratify argument to keep the proportion of the target variable `survival` equal in all sets. This is important because we want to train and test the model on roughly the same proportion of the target variable to properly train the model and get meaningful evaluation metrics.

# %%
# First split: training vs other (val+test)
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3,  # 70% training, 30% for val+test
    stratify=y,     # maintain class distribution between train and val+test
    random_state=SEED
)

# Second split: split remaining data into val and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,  # split the 30% into 15% val, 15% test
    stratify=y_temp, # maintain class distribution between val and test
    random_state=SEED
)

# Verify split sizes
print(f"Training set: {len(X_train)} ({len(X_train)/len(X):.1%})")
print(f"Validation set: {len(X_val)} ({len(X_val)/len(X):.1%})")
print(f"Test set: {len(X_test)} ({len(X_test)/len(X):.1%})")


# %% [markdown]
# We can see that the split was done correctly. Now we have to check if the distribution of the attributes was maintained in each dataset.

# %%
def compare_distributions(train, val, test, feature):
    # Skip if feature is not numeric
    if not np.issubdtype(train[feature].dtype, np.number):
        print(f"\nSkipping {feature} - not numeric")
        return
        
    print(f"\n{feature} Statistics:")
    print(f"{'Set':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    print(f"Train: {train[feature].mean():10.3f} {train[feature].std():10.3f} {train[feature].min():10.3f} {train[feature].max():10.3f}")
    print(f"Val:   {val[feature].mean():10.3f} {val[feature].std():10.3f} {val[feature].min():10.3f} {val[feature].max():10.3f}")
    print(f"Test:  {test[feature].mean():10.3f} {test[feature].std():10.3f} {test[feature].min():10.3f} {test[feature].max():10.3f}")

# Check each numerical feature
numeric_features = X_train.select_dtypes(include=[np.number]).columns
for feature in numeric_features:
    compare_distributions(X_train, X_val, X_test, feature)

# %% [markdown]
# As we can see, the distribution of the attributes is roughly the same across the training, validation, and test datasets. This is important because we want our model to generalize well, so we want to train it with the same distribution as the data it well get at test time. Directly, this also affects the reliability of the evaluation metrics we use to tune the model in validation.

# %% [markdown]
# Now I want to check the target distribution across the different datasets. It should be equal because of the `stratify` argument we used, but I want to see if it is unbalanced.

# %%
def check_target_distribution(y_train, y_val, y_test):
    print("\nOriginal Class Distribution:")
    print(f"Died: {(y==0).mean():.1%}")
    print(f"Survived: {(y==1).mean():.1%}")
    print("\nTarget Distribution:")
    print(f"{'Set':<6} {'Died':>10} {'Survived':>10}")
    print("-" * 30)
    print(f"Train: {(y_train==0).mean():10.3f} {(y_train==1).mean():10.3f}")
    print(f"Val:   {(y_val==0).mean():10.3f} {(y_val==1).mean():10.3f}")
    print(f"Test:  {(y_test==0).mean():10.3f} {(y_test==1).mean():10.3f}")

check_target_distribution(y_train, y_val, y_test)

# %% [markdown]
# # Handling Missing Values (After Split)
# For imputing the missing values after splitting the data, we have to choose our imputatation strategy based on the training set and then implement that strategy seperately for the validation and test datasets.
# 
# This way, we use the same strategy to impute values, which is good because we want to keep our data distribution consistent, but we only use each dataset in isolation, avoiding the risk of data leakage through computing training missing values with test and validation data.

# %% [markdown]
# ## Missing values in `fare`
# There is only one missing value in the `fare` attribute.

# %%
print("Training set null values:")
print(X_train.loc[X_train['fare'].isnull()])
print("\nValidation set null values:")
print(X_val.loc[X_val['fare'].isnull()])
print("\nTest set null values:") 
print(X_test.loc[X_test['fare'].isnull()])

# %% [markdown]
# This passenger is a **60.5 year** old **man** (no siblings/spouses or parents/children) in **3rd class** who embarked from **S**. We use this information to find passengers with the same characteristics and use the mean fare of these passengers to impute the missing value.
# 

# %%
# Calculate mean fare for each dataset separately
for dataset in [X_train, X_val, X_test]:
    mask = (
        (dataset['pclass'] == 3) & 
        (dataset['sex'] == 'male') & 
        (dataset['sibsp'] == 0) & 
        (dataset['parch'] == 0) & 
        (dataset['embarked'] == 'S') &
        (dataset['age'] > 60)
    )
    mean_fare = dataset[mask]['fare'].mean()
    
    # If no matching passengers found, use overall mean for that dataset
    if pd.isna(mean_fare):
        mean_fare = dataset['fare'].mean()
        
    print(f"\nMean fare for dataset: {mean_fare:.2f}")
    print("\nMatching passengers:")
    print(dataset[mask][['age', 'fare']])
    
    # Impute missing values
    dataset.loc[dataset['fare'].isnull(), 'fare'] = mean_fare

# %% [markdown]
# In this case, all the passengers found with common characteristics are in the training set, so there is no risk of data leakage.

# %% [markdown]
# ## Missing values in `embarked`
# There are 2 missing values in the `embarked` attribute.
# 
# 

# %%
print("Training set null values:")
print(X_train.loc[X_train['embarked'].isnull()])
print("\nValidation set null values:")
print(X_val.loc[X_val['embarked'].isnull()])
print("\nTest set null values:")
print(X_test.loc[X_test['embarked'].isnull()])

# %% [markdown]
# Looking at the 2 records, we see that they are both in **first class**, with no siblings/spouses or parents/children, the same **fare price of 80**, the **same ticket number**, and the **same cabin**. We can safely assume that they embarked from the same port.
# 
# To find which port they embarked from, I will group `embarked` by `pclass` and calculate the mean for each class at each port to see which one they are more close to. This will allow me to make a more informed choice for the imputation, as opposed to using a common statistic.

# %%
for dataset in [X_train, X_val, X_test]:
    print(f"\nDataset fare means by embarked and pclass:")
    print(dataset.groupby(['embarked', 'pclass'])['fare'].mean())


# %% [markdown]
# The closest price range to the missing value is port 'S' (fare in first class), so that is what I will use to impute these values.

# %%
for dataset in [X_train, X_val, X_test]:
    dataset.loc[dataset['embarked'].isnull(), 'embarked'] = 'S'


# %% [markdown]
# In this very specific case (of how the data is split), there is no risk of leakage because both these records are in validation and test data, so there is nothing learned in the training dataset. Yet, it is still, in my opinion, a clever way to impute the data.

# %% [markdown]
# In the [notebook online](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial) referenced ealier, the author uses Wikipedia of the titanic page as an indirect data source to identify the people who's `embarked` value was missing which gave him/her 100% confidence in the imputed value. Sadly, I wasn't clever enough to think of that, but I got close.

# %% [markdown]
# ## Missing values in `age`

# %% [markdown]
# The simplest method for dealing with missing values is just dropping them. However, we can't just drop the missing values in the `age` attribute because it would be a significant loss of data and hinder our model training.
# 
# There are several ways to impute the missing values in the `age` attribute. We can use the median age or the mean age. However, for each of these methods, it is probably better to compute the corresponding value for each class of another attribute, such as `pclass` and/or `sex` as this would be a more likely value for the missing value, as opposed to the median or mean of the age in the whole dataset. This would help us capture a more nuanced relationship between age and other attributes, which could be lost in a more crude method.
# 
# Other methods of data impution like a middle of the range (with the idea to not impact the distribution of the attribute, or not to add meaning) or out of range replacement (e.g. a -1 with the idea that the model learns that those values were missing or different) would lose information compared to the median or mean.
# 
# The classes we should consider are the `pclass`, `embarked`, and `sex` attributes. My logic is that the age distribution of males and females could be different. Similarly, the age of each passenger class could be different because the young passengers could be saving money by being in the lower class, and the old passengers might be wealthy and choose to spend more money on the higher class. Finally, the age statistic we use could have a different distribution in each port of embarkation. One could be a young student town and the other could be a wealthy old town.
# 
# To explore these differences, I will explore the mean, median, and count of the `age` attribute by each class of `pclass`, `sex`, and `embarked`.

# %%
# Training data
print("Training data:")
print(X_train.groupby('pclass')['age'].agg(['mean', 'median', 'count']))

# Validation data 
print("\nValidation data:")
print(X_val.groupby('pclass')['age'].agg(['mean', 'median', 'count']))

# Test data
print("\nTest data:")
print(X_test.groupby('pclass')['age'].agg(['mean', 'median', 'count']))

# %%
# Training data
print("Training data:")
print(X_train.groupby('sex')['age'].agg(['mean', 'median', 'count']))

# Validation data
print("\nValidation data:")
print(X_val.groupby('sex')['age'].agg(['mean', 'median', 'count']))

# Test data
print("\nTest data:")
print(X_test.groupby('sex')['age'].agg(['mean', 'median', 'count']))


# %%
# Training data
print("Training data:")
print(X_train.groupby('embarked')['age'].agg(['mean', 'median', 'count']))

# Validation data
print("\nValidation data:")
print(X_val.groupby('embarked')['age'].agg(['mean', 'median', 'count']))

# Test data
print("\nTest data:")
print(X_test.groupby('embarked')['age'].agg(['mean', 'median', 'count']))

# %% [markdown]
# It is evident that there is a difference in the mean or median age for different classes of `pclass` and `sex`. However, the difference in the mean or median age for each port of embarkation is not as significant. So we will use the median age for each class of `pclass` and `sex` to impute the missing values in the `age` attribute.

# %% [markdown]
# I think the median age is an accurate and probable representation of the age of the passengers in each class and sex. I chose the median over the mean because there is some level of skewness in the distribution of age and median gives a better measure of central tendency in these cases, though there probably isn't a huge difference in this case.

# %%
# Training data
plt.figure(figsize=(10, 6))
plt.hist(X_train['age'], bins=30, edgecolor='black')
plt.title('Age Distribution - Training Set')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()

# Validation data 
plt.figure(figsize=(10, 6))
plt.hist(X_val['age'], bins=30, edgecolor='black')
plt.title('Age Distribution - Validation Set')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()

# Test data
plt.figure(figsize=(10, 6))
plt.hist(X_test['age'], bins=30, edgecolor='black')
plt.title('Age Distribution - Test Set')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()


# %%
# Training data
X_train['age'] = X_train.groupby(['pclass', 'sex'])['age'].transform(lambda r: r.fillna(r.median()))

# Validation data
X_val['age'] = X_val.groupby(['pclass', 'sex'])['age'].transform(lambda r: r.fillna(r.median()))

# Test data 
X_test['age'] = X_test.groupby(['pclass', 'sex'])['age'].transform(lambda r: r.fillna(r.median()))

# Verify no missing values remain
print("Missing values in training set:", X_train['age'].isnull().sum())
print("Missing values in validation set:", X_val['age'].isnull().sum()) 
print("Missing values in test set:", X_test['age'].isnull().sum())

# %% [markdown]
# ## Missing values in `cabin`
# First, I want to see if I can deduce anything from the `cabin` attribute that will help me impute the missing values.
# 
# The obvious thing to do is to extract the deck letter from the `cabin` attribute. After that, I want to compute statistics of different attributes for each deck to see if there is a certain insight that can help me synthesize informative features.

# %%
# Training data
X_train['deck'] = X_train['cabin'].str[0]
train_deck_stats = pd.concat([X_train, y_train], axis=1).groupby('deck').agg({
    'survived': ['mean', 'count']
}).round(3)
train_deck_stats.columns = ['Survival Rate', 'Count']
X_train = X_train.drop('cabin', axis=1)

# Validation data
X_val['deck'] = X_val['cabin'].str[0]
val_deck_stats = pd.concat([X_val, y_val], axis=1).groupby('deck').agg({
    'survived': ['mean', 'count']
}).round(3)
val_deck_stats.columns = ['Survival Rate', 'Count']
X_val = X_val.drop('cabin', axis=1)

# Test data
X_test['deck'] = X_test['cabin'].str[0]
test_deck_stats = pd.concat([X_test, y_test], axis=1).groupby('deck').agg({
    'survived': ['mean', 'count']
}).round(3)
test_deck_stats.columns = ['Survival Rate', 'Count']
X_test = X_test.drop('cabin', axis=1)

print("\nDeck Statistics - Training Set:")
print(train_deck_stats)
print("\nDeck Statistics - Validation Set:")
print(val_deck_stats)
print("\nDeck Statistics - Test Set:") 
print(test_deck_stats)


# %% [markdown]
# This shows that the `deck` and the attributes `pclass`, `fare`, and `age` have some relationship, or aren't totally unrelated. I want to see if I can use a Clustering algorithm like KMeans Clustering to group the passengers into different decks. This is a commonly used approach.

# %%
"""
# Fit clustering model on training data only
clustering_features = ['pclass', 'fare', 'age']

# Scale training data
scaler = StandardScaler()
train_cluster_data = scaler.fit_transform(X_train[clustering_features])

# Fit KMeans on training data
kmeans = KMeans(n_clusters=4, random_state=SEED)
kmeans.fit(train_cluster_data)

# Create cluster mapping from training data
cluster_deck_map = {}
X_train['cabin_cluster'] = kmeans.predict(train_cluster_data)
for cluster in range(4):
    mask = X_train['cabin_cluster'] == cluster
    most_common_deck = X_train[mask]['deck'].mode().iloc[0] if not X_train[mask]['deck'].isna().all() else 'Unknown'
    cluster_deck_map[cluster] = most_common_deck

# Apply clustering to all datasets
for dataset in [X_train, X_val, X_test]:
    # Scale using training scaler
    scaled_data = scaler.transform(dataset[clustering_features])
    # Predict clusters using trained model
    dataset['cabin_cluster'] = kmeans.predict(scaled_data)
    # Map clusters to decks
    dataset['deck'] = dataset.apply(
        lambda row: cluster_deck_map[row['cabin_cluster']] if pd.isna(row['deck']) else row['deck'],
        axis=1
    )
    dataset.drop('cabin_cluster', axis=1, inplace=True)
"""

# %% [markdown]
# I tried to use KMeans to group similar data into clusters and assign the deck of the deck cluster they've been assigned to.

# %% [markdown]
# After setting the `n_clusters` hyperparameter to 4 arbitrarily, I performed hyperparameter tuning of the KMeans clustering using Grid Search. However, the accuracy of the model decreased on validation. I believe this is because optimizing this KMeans doesn't translate directly to more accurate inference of missing values for the `deck` attribute. One reason, for example, is that the mathematical optimization of this algorithm doesn't take the positions and nature of the cabins/decks in the ship into account.

# %% [markdown]
# Another approach could be te use the K-Nearest Neighbors algorithm to predict the missing values based on the existing data. This would be an example of training a model to use its predictions to fill missing values for another model we want to train.

# %%
clustering_features = ['pclass', 'fare', 'age']

# Fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[clustering_features])
X_val_scaled = scaler.transform(X_val[clustering_features]) 
X_test_scaled = scaler.transform(X_test[clustering_features])

# Train KNN on training data
known_deck_mask_train = ~X_train['deck'].isna()
X_known_train = X_train_scaled[known_deck_mask_train]
y_known_train = X_train.loc[known_deck_mask_train, 'deck']

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_known_train, y_known_train)

# Predict missing decks for each dataset
for dataset, scaled_data in [(X_train, X_train_scaled), 
                            (X_val, X_val_scaled),
                            (X_test, X_test_scaled)]:
    unknown_deck_mask = dataset['deck'].isna()
    X_unknown = scaled_data[unknown_deck_mask]
    if len(X_unknown) > 0:
        predicted_decks = knn.predict(X_unknown)
        dataset.loc[unknown_deck_mask, 'deck'] = predicted_decks

X_train.head()
X_test.head()

# %% [markdown]
# I did not go deep into hyperparameter tuning in this stage, but using KNN for imputing `deck` resulted in a slightly better performance of the model on the validation dataset.
# 
# My intuition is that KNN is, in general, better for this task than KMeans clustering for a couple of reasons:
# 1. KNN makes a prediction for the missing decks based on closest neighbors. KMeans groups data into clusters then assigns the most common deck to the missing deck. KMeans has an extra (indirect)  step (choosing most common in a cluster) which could lead to more error.
# 2. The hyperparamater we can set for KNN serves us more than the one we set for KMeans. For KNN, we choose the number of nearest neighbors to consider, which is direct and makes sense. For KMeans, we have to choose the number of clusters which are more indirect and we don't really know. I was mistakenly tempted to set `n_clusters` to 7 because there are 7 decks, but I then realized clustiring is based on how the data naturally groups up considering the clustering attributes like `pclass`, `fare`, and `age` in this case, not the fact that we want to force them into 7 clusters.

# %% [markdown]
# It is crucial to perform this imputation after splitting the data and to perform it separately for each dataset. The reason is that an algorithm like KNN or KMeans uses the dataset and makes a prediction based on the distance between the items.
# 
# If we were to perform this step before the split, the imputed values would be determine based on the entire dataset (including items that would later be split into validation and test datasets). This would be a big case of data leakage since the training data would know some information because of this from the validation and training datasets.

# %% [markdown]
# # Discretizing Features
# Instead of treating `age` and `fare` attributes as continuous attributes, it might be beneficial to divide them into sections (bins), making them a discrete attribute. This has a couple of benefits:
# - Enables our Linear Regression model to capture some non-linear relationship between these attributes and survival. For example, instead of learning from age as a continuous model, it can learn the relationship between age groups like children, adults, seniors and survival, which might be helpful.
# - A potential solution to outliers: When we do the binning, the outliers fall into the first and last bins, still carrying the information that they are on the end of the spectrum, but not impacting the distance learned by LogisticRegression.
# - As a result, there is reduced overfitting to outliers and variance in these attributes
# 
# For discretizing, we would group these attributes into bins. My thought was to use general age divide between age classes (child, teenager, adult, senior, etc), and quartiles for fare.
# 
# After that, we would use one-hot encoding to encode these attributes the same way we would for other categorical variables. My initial thought was that Label Encoding could be helpful because there is a natural order to these attributes between these bins, but upon further research, I found out that a label-encoder would imply and artificial linear relationship between the bins to the LinearRegression model, but it could be a valid choice for a model like Decision Tree where it could learn the threshold for each bin.
# 
# **Note**: I tried discretizing in another branch, but it lowered my model accuracy in the end so I didn't implement it in this notebook submission.

# %% [markdown]
# # Encoding Categorical Variables
# The following are the categorical variables in the dataset:
# - `pclass` (1,2,3)
# - `parch` (numbers)
# - `sibsp` (numbers)
# - `ticket_count` (numbers)
# - `sex` (male, female)
# - `embarked` (C, Q, S)
# - `deck` (A, B, C, D, E, F, G, T)
# - `title` (Mr., Mrs., Miss., Master., Rare)
# - `home.dest` (A lot of values, mostly "Unknown" and unique one-offs)
# 
# `pclass` is an interesting case because it is categorical, in the sense that it's not a continuous number. But the number (1,2,3) for the class conveys the information that the class is higher or lower, which can be important. Therefore, I will not encode it and leave it for the scaling stage. Same thing applies for `sibsp`, `parch`, and `ticket_count`.
# 
# `home.dest` is also special because I want to encode it in a way that avoids high dimensionality.

# %% [markdown]
# For handling `home.dest`, I remember that there was a very long tail of values.

# %% [markdown]
# The distribution of `home.dest` is such that 520 (around half) of the data is unknown. Then, there are 4 items with frequency of greater than 10: 
# - New York, NY - 70
# - London - 14
# - Paris, France - 11
# - Montreal , PQ - 11
# 
# The rest is the "long tail" of the distribution with rare values.
# 
# Therefore, I will try one-hot encoding for these values, "Unknown", and group everything else into a "other" category. 
# 
# I have doubts about this feature since most of it is unknown and it probably isn't very informative to begin with. It might introduce some overfitting into the model. I will think more about this in the feature selection stage.

# %% [markdown]
# There are differet encoders we could use for encoding. The most common and simple one is one-hot encoding, which I will use. 
# Another option would be to use mean encoding, but that is usually used for high-cardinality features. So we will stick with the simpler one-hot encoding.

# %%
# Initialize encoders
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

# Replace the long tail of home.dest with "Other" for all datasets
major_destinations = ['New York, NY', 'London', 'Paris, France', 'Montreal, PQ', 'Unknown']
for dataset in [X_train, X_val, X_test]:
    dataset['home.dest'] = dataset['home.dest'].apply(
        lambda x: x if x in major_destinations else 'Other'
    )

# One-hot encode other categorical variables - fit on training data only
categorical_cols = ['embarked', 'deck', 'home.dest', 'title', 'sex']

encoded_train = onehot_encoder.fit_transform(X_train[categorical_cols])
encoded_val = onehot_encoder.transform(X_val[categorical_cols])
encoded_test = onehot_encoder.transform(X_test[categorical_cols])

# Get feature names after encoding
feature_names = []
for i, col in enumerate(categorical_cols):
    cats = onehot_encoder.categories_[i][1:]
    feature_names.extend([f"{col}_{cat}" for cat in cats])

# Create encoded DataFrames
encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=X_train.index)
encoded_val_df = pd.DataFrame(encoded_val, columns=feature_names, index=X_val.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=X_test.index)

# Drop original columns and concatenate encoded ones
X_train = pd.concat([X_train.drop(categorical_cols, axis=1), encoded_train_df], axis=1)
X_val = pd.concat([X_val.drop(categorical_cols, axis=1), encoded_val_df], axis=1)
X_test = pd.concat([X_test.drop(categorical_cols, axis=1), encoded_test_df], axis=1)

# Verify the encoding worked
print("Final columns:", X_train.columns.tolist())
print("\nShape of encoded datasets:")
print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

X_train.head()

# %% [markdown]
# - The `sex` attribute has been one-hot encoded into `sex_male`.
# - `embarked` is now one-hot encoded into `embarked_Q` and `embarked_S`.
# - `deck` is now one-hot encoded into `deck_B`, `deck_C`, `deck_D`, `deck_E`, `deck_F`, `deck_G`, and `deck_T`.
# - `home.dest` is now one-hot encoded into `home.dest_New York, NY`, `home.dest_Unknown`, `home.dest_Paris, France`, `home.dest_Montreal, PQ`, and `home.dest_Other`.
# - `title` is now one-hot encoded into `title_Mr`, `title_Miss`, `title_Mrs`, and `title_Rare`.
# 
# Encoding categorical variables is necessary for some algorithms like LogisticRegression because they require numerical input and can't use categorical input. We use one-hot encoding to convert these categorical variables into a binary n-1 dimensional feature as we use the `drop='first'` argument to drop the first category of each attribute, since it can be inferred from the others.
# 
# By converting categorical features into numerical ones, the Machine Learning models like Linear Regression can perform all the mathematical operations, distance calculations, regularization, feature importance calculation, and calculating weights which would not have been possible with categorical methods. This directly allows and benefits model accuracy, feature selection, and model interpretability.
# 
# There's also an important issue of **category consistency**. Since we are performing the encoding after the data split, there is a possibility that a category of a categorical attribute is present in the training set but not the holdout sets, or vice versa. Our encoder fits on the training data and performs that transformation on the validation and test data. This means that if there is a category in the training dataset that doesn't appear in the holdout sets, the one-hot encoded column for it still exists for the holdout set, but every value is 0. This is good because we want the structure of our dataset to be consistent. The alternative would be if that column wouldn't exist for the holdout, which would create a conflict between the data structure later. 
# Conversely, if a category is present in the holdout set and not in the training, we would get an error by default, but we use the `handle_unknown='ignore'` argument to ignore that column and not create it in the holdout set. This isn't problematic because we didn't get any meaningful information from that column to train the model.
# 

# %% [markdown]
# # Scaling Features
# In this stage, I will scale the features using the `StandardScaler` and `MinMaxScaler` classes.

# %% [markdown]
# ## Standadizing
# Standardizing is when we rescale the feature values to have the properties of a standard normal distribution with a sample mean of 0 and a sample standard deviation of 1. It is useful when the data follows a Gaussian distribution or when the algorithm assumes a normal distribution.
# 
# $$\hat{x}^{(j)} \leftarrow \frac{x^{(j)} - \mu^{(j)}}{\sigma^{(j)}}$$
# 
# 

# %% [markdown]
# ## Normalizing
# Normalizing is when we rescale the feature values to a certain range, most commonly between 0 and 1 or -1 and 1.It is useful when the data does not follow a Gaussian distribution or when the algorithm is sensitive to the scale of the data.
# 
# $$\hat{x}^{(j)} \leftarrow \frac{x^{(j)} - \min(x^{(j)})}{\max(x^{(j)}) - \min(x^{(j)})}$$

# %% [markdown]
# To find out whether we should standardize or normalize a certain feature, we can use the Shapiro-Wilk test to see if the feature follows a Gaussian distribution, in which case we should probably standardize the data.
# 
# If the p-value is less than the significance level of 0.05, we conclude that the data is not normally distributed. Otherwise, we conclude that the data is normally distributed.
# 
# For example, we will use this for the `pclass` attribute. A priori, since `pclass` is a categorical attribute of the class, my intuition says that it doesn't follow a normal distribution and we probably have to normalize it rather than standardize it. I will test my intuition with the Shapiro-Wilk test.

# %%
stat, p = stats.shapiro(X_train['pclass'])
stat,p

# %% [markdown]
# The p-value for this test of `pclass` attribute is very close to zero, concluding that it doesn't follow a normal distribution. Therefore, it is better to normalize to a range between 0 and 1.
# I will now conduct this test for other numerical attributes.

# %%
stat, p = stats.shapiro(X_train['pclass'])
stat,p

# %% [markdown]
# I will now run this test for all other numerical variables: `age`, `sibsp`, `parch`, `fare`, `ticket_count`.

# %%
for attribute in ['age', 'sibsp', 'parch', 'fare', 'ticket_count']:
    stat, p = stats.shapiro(X_train[attribute])
    print("p-value for " + attribute + "= "+str(p))
    if p <= 0.05:
        print("Therefore, "+attribute+" dosen't follow a normal distribution.")
    else:
        print("Therefore, "+attribute+" follows a normal distribution.")

# %% [markdown]
# After running the Shapiro-Wilk test, I see that none of the attributes follow the normal distribution under this test. However, it is quite rare to find perfect normal distribution in any real-world scenario. Therefore, I will go with my intuition: continuous attributes like age and fare, even if skewed, would benefit more from standardization to make sure they're not dominant in the model. Discrete attributes like pclass, sibsp, parch, and ticket count would benefit more from normalization.
# 
# The main difference is between the detection, handling, and information preservation about outliers. There are outliers in `age` and `fare` that are meaningful. Therefore, standardization allows us to keep that information while reducing feature dominance. The categories in the other attributes are all valid and "outliers" isn't a meaningful concept in them. Therefore, normalization is a better choice. 
# 
# In general, the purpose of this scaling is to avoid feature dominance (e.g. max fare is 500, max age is 80, but we don't want the model to think fare is more important than age), equal impact of regularization on each attribute (the regularization constant would have the equal effect on all weights) and better model interpretibility (weights show us the relative importance of each feature). Apart from that, it is an assumption that a lot of algorithms make. This is why we had to scale data for using KMeans and KNN to impute `deck` earlier.

# %% [markdown]
# We use the `fit_transform` method of the scalers on the training data and the `transform` method on the test and validation data. We do this to use the scaling parameters learned from the training data to scale the validation and test data. This will prevent us from manually changing the distribution between training and holdout sets. Furthermore, it also helps in preventing potential leakage that would convey distribution information about the test data to the training data.

# %%
continuous = ['age', 'fare']
discrete = ['pclass', 'sibsp', 'parch', 'ticket_count']

# Initialize scalers
normalizer = MinMaxScaler()
standardizer = StandardScaler()

#robust_scaler = RobustScaler()
#titanic_data['fare'] = robust_scaler.fit_transform(titanic_data['fare'].values.reshape(-1, 1)) // An attempt to make fare more robust to outliers, but it didn't make a difference in the model's performance

# Fit and transform on training data
X_train[continuous] = standardizer.fit_transform(X_train[continuous])
X_train[discrete] = normalizer.fit_transform(X_train[discrete])

# Transform validation data using fitted scalers
X_val[continuous] = standardizer.transform(X_val[continuous]) 
X_val[discrete] = normalizer.transform(X_val[discrete])

# Transform test data using fitted scalers
X_test[continuous] = standardizer.transform(X_test[continuous])
X_test[discrete] = normalizer.transform(X_test[discrete])

X_train.head()

# %% [markdown]
# # Addressing Class Imbalance
# As we saw in the previous stage, we have almost twice as much deaths than survivals. We have to fix this class imbalance to have a more even dataset. There are mainly two ways of addressing this issue: undersampling and oversampling.
# 
# Undersampling is where we remove examples in the over-represented class to get it closser to the minority class. The problem with this method that we can lose a lot of valuable data. Especially in this case where we don't have too many records to begin with, we cannot afford to do undersampling. Therefore we have to go for oversampling.
# 
# Oversampling is where we try to balance the classes by creating synthetic examples from the under-represented class.
# 
# The two most used oversampling techniques are SMOTE and ADASYN. They work similarly but ADASYN can produce improved results. This is because ADASYN caluclates the density of each minority-class example using k-nearest neighbors and assigns a weight to that example based on the calculated density. Then, when generating synthetic records, ADASYN uses those weights to determine the number of synthetic examples to generate for each example. This enables ADASYN to capture more complex relationships and introduce less noise compared to SMOTE. The drawback is that it is more computationally-intensive. But, since we have a small dataset, that is not a big problem. So I will use ADASYN.

# %% [markdown]
# Another valid approach would be to use a hybrid strategy of ADASYN (Oversampling) + Tomek Links (Undersampling).
# 
# 1. ADASYN would oversample the minority class (survivors)
# 2. Tomek Links would remove pairs of examples from opposite classes that are nearest neighbors of each other.
# 
# This could be beneficial since it would remove borderline or noisy examples that could confuse the classifier, improving the decision boundary and reducing overfitting from synthetic examples.
# 
# However, it would remove some real data points and not offer a huge improvement since the dataset is small and relatively clean. (**Note from future**: Model is tested and is already performing well so ADASYN + Tomek Links would have added unnecessary complexity)

# %% [markdown]
# We always address the data imbalance after splitting the data because we want to avoid changing the distribution of our test and validation data because they are supposed to represent what we will take as input in production, which is the same as the original dataset. Furthermore, if we decide to go for an oversampling strategy that creates synthetic data, the algorithm would consider data that would be later allocated to validation and test data for creating the synthetic data, introducing data leakage.

# %%
# Apply ADASYN only to training data
adasyn = ADASYN(random_state=SEED) # n_neighbors = 5 default, could test later.
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

# Check class distribution before and after
print("Before ADASYN:")
print(f"Class 0: {(y_train==0).sum()} ({(y_train==0).mean():.1%})")
print(f"Class 1: {(y_train==1).sum()} ({(y_train==1).mean():.1%})")

print("\nAfter ADASYN:")
print(f"Class 0: {(y_train_balanced==0).sum()} ({(y_train_balanced==0).mean():.1%})")
print(f"Class 1: {(y_train_balanced==1).sum()} ({(y_train_balanced==1).mean():.1%})")

print(f"{(y_train_balanced==1).sum() - (y_train==1).sum()} synthetic examples generated.")

# %%
# Plot class distribution before and after ADASYN
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.pie([sum(y_train==0), sum(y_train==1)], labels=['Deaths', 'Survivals'], autopct='%1.1f%%')
plt.title('Before ADASYN')

plt.subplot(1, 2, 2)
plt.pie([sum(y_train_balanced==0), sum(y_train_balanced==1)], labels=['Deaths', 'Survivals'], autopct='%1.1f%%')
plt.title('After ADASYN')

plt.tight_layout()
plt.show()


# %% [markdown]
# As we can see, ADASYN generated synthetic examples and balanced the training the dataset from 38-62 to 49-51. This enables our learning algorithm later to make better predictions by weighing different types of mistakes (false positives and false negatives) equally.

# %% [markdown]
# ## Feature Selection
# 
# It is not smart to include all our features in training the model for various reasons:
# - By including more features, we increase the dimensionality of our dataset which could result in a more sparse dataset, especially because of the one-hot encoded features, and increase computational costs.
# - Using a simple shallow algorithm like LogisticRegression, we are prone to overfitting to noise with more features that we have. We have to try to include only the most important and predictive features.
# - High number of features makes model interpretability more difficult.
# 
# We have already eliminatied certain features from our dataset for various reasons:
# - `boat` and `body`: introduced leakege since they were directly related to survival, which is our target variable, and were from the future (wouldn't be known at test time).
# - `name`: doesn't convey useful information, any attempt to encode it would result in a significatnly large-dimension and sparse feature matrix, and would introduce the possibility of noise and overfitting to names, though we extracted the `title` feature from it.
# - `ticket` for reasons similar to `name`, though we extracted the `ticket_count` from it.
# 
# At this point, our features are:
# - `pclass`
# - `sibsp`
# - `parch`
# - `sex`
# - `age`
# - `ticket_count`
# - `deck`
# - `home.dest`
# - `title`
# - `embarked`
# 
# 

# %% [markdown]
# ## Removing Low-Variance High-Correlation Features
# 
# We want to remove features with low variance and high correlation. The logic is that low-variance features don't convey much infomation because they don't vary through a dataset, so the model won't learn much from them.
# 
# Absence of multicollinearity is a precondition of using Logistic Regression because it makes it hard to distinguish the individual impact of features on the model, therefore we want to remove high-correlation feature pairs.

# %%
def select_features(X, y, corr_pairs):
    if not corr_pairs:
        return X.columns.tolist()
    
    features_to_drop = set()
    for pair in corr_pairs:
        f1, f2 = pair
        corr1 = abs(np.corrcoef(X[f1], y)[0,1])
        corr2 = abs(np.corrcoef(X[f2], y)[0,1])
        features_to_drop.add(f1 if corr1 < corr2 else f2)
    
    return [f for f in X.columns if f not in features_to_drop]

# First, identify one-hot encoded columns
ohe_prefixes = ['embarked_', 'deck_', 'home.dest_', 'title_', 'sex_']
ohe_cols = [col for col in X_train_balanced.columns if any(col.startswith(prefix) for prefix in ohe_prefixes)]
numeric_cols = [col for col in X_train_balanced.columns if col not in ohe_cols]

# Apply variance threshold only to numeric columns
print(f"Original numeric columns: {numeric_cols}")
print(f"Original shape: {X_train_balanced[numeric_cols].shape}")

selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train_balanced[numeric_cols])

# Get variances for each feature
variances = selector.variances_
for col, var in zip(numeric_cols, variances):
    print(f"Variance for {col}: {var:.4f}")

kept_features = X_train_balanced[numeric_cols].columns[selector.get_support()]
removed_features = set(numeric_cols) - set(kept_features)

print(f"\nRemoved features (variance < 0.01): {removed_features}")
print(f"Kept numeric features: {kept_features}")

variance_features = list(kept_features) + ohe_cols
print(f"\nFinal feature set including OHE cols: {len(variance_features)} features")

# Check correlation only between numeric features
correlation_matrix = X_train_balanced[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

def get_redundant_pairs(corr_matrix, threshold=0.8):
    pairs_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                pairs_to_drop.add((corr_matrix.columns[i], corr_matrix.columns[j]))
    return pairs_to_drop

high_corr_pairs = get_redundant_pairs(correlation_matrix)
print("\nHighly correlated feature pairs:")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {correlation_matrix.loc[pair[0], pair[1]]:.3f}")

# Keep one feature from each highly correlated pair
final_numeric_features = select_features(X_train_balanced[numeric_cols], y_train_balanced, high_corr_pairs)
final_features = final_numeric_features + ohe_cols

# Update datasets
X_train_balanced = X_train_balanced[final_features]
X_val = X_val[final_features]
X_test = X_test[final_features]

# %% [markdown]
# The `parch` feature was disqualified because of low variance. No features were disqualified because of high correlation.

# %% [markdown]
# ## Automated Feature Selection

# %% [markdown]
# A more unified approach to feature selection is to use feature selection algorithms that perform more complex operations to find important features. In this notebook, I will try Boruta, Lasso, and Mututal Information for feature selection.

# %% [markdown]
# ### Boruta
# Boruta uses random forests to compare each feature to its shuffled version to compare the importance of the feature to random chance.
# 
# The features with low rank (more importance) can be understood to have more predictive power since they significantly outperform the randomized, shuffled version.

# %%
# Initialize Boruta
rf = RandomForestClassifier(n_jobs=-1, random_state=SEED) # n_jobs=-1 to use all available CPU cores
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=SEED) #n_estimators='auto' to use the default number of trees

# Fit on balanced training data
boruta.fit(X_train_balanced.values, y_train_balanced)

# Get selected features
selected_features = X_train_balanced.columns[boruta.ranking_ <= 4]
print("\nSelected features:")
for feature, rank in zip(X_train_balanced.columns, boruta.ranking_):
    print(f"{feature}: {'Selected' if rank <= 4 else 'Rejected'} (Rank: {rank})")
# Sort by Boruta rank
boruta_importance = pd.DataFrame({
    'Feature': X_train_balanced.columns,
    'Boruta_Rank': boruta.ranking_
}).sort_values('Boruta_Rank')

print("\nFeatures sorted by Boruta rank:")
print(boruta_importance)


# %% [markdown]
# ### Lasso
# Lasso (l1) is a regularization method to reducte the variance of the model. How it works is by creating coefficients that will force less important features to be zero. By comparing these coefficients. We can identify the most important features that contribute to the prediction.

# %%
# Since we already scaled the data, we don't need to scale it again for Lasso.

# Fit LassoCV (automatically finds best alpha through CV)
lasso = LassoCV(
    cv=5, 
    random_state=SEED,
    max_iter=2000,
    n_jobs=-1
)
lasso.fit(X_train_balanced, y_train_balanced)

# Get feature importance from Lasso
lasso_importance = pd.DataFrame({
    'Feature': X_train_balanced.columns,
    'Lasso_Coef': np.abs(lasso.coef_),
    'Boruta_Rank': boruta.ranking_
})

# Sort by absolute coefficient value
lasso_importance = lasso_importance.sort_values('Lasso_Coef', ascending=False)

print("Feature Importance Comparison:")
print(lasso_importance)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.bar(range(len(lasso_importance)), lasso_importance['Lasso_Coef'])
plt.xticks(range(len(lasso_importance)), lasso_importance['Feature'], rotation=45, ha='right')
plt.title('Lasso Feature Importance')
plt.tight_layout()
plt.show()

# %% [markdown]
# The `parch` was disqualified because of its low variance, however Lasso is showing that it has considerable feature importance.
# 
# Lasso is a good feature selection method that indicates which features have high predictive power. Therefore, this means that `parch` varies little but that doesn't necessarily mean it's not important and doesn't predict survival. Lasso tells us that it is, in fact, important. Therefore, I will include `parch` in my final feature selection.

# %% [markdown]
# ### Mutual Information
# Mutual information tells us how much insight each feature gives regarding the target variable. We can use this to help us decide which features are more important.

# %%
# Calculate information gain for each feature
info_gains = mutual_info_classif(X_train_balanced, y_train_balanced)

# Create dataframe of features and their information gain scores
info_gain_importance = pd.DataFrame({
    'Feature': X_train_balanced.columns,
    'Info_Gain': info_gains,
    'Lasso_Coef': lasso_importance['Lasso_Coef'].values,
    'Boruta_Rank': lasso_importance['Boruta_Rank'].values
})

# Sort by information gain
info_gain_importance = info_gain_importance.sort_values('Info_Gain', ascending=False)

print("\nFeature Importance by Information Gain:")
print(info_gain_importance)

# Plot information gain scores
plt.figure(figsize=(12, 6))
plt.bar(range(len(info_gain_importance)), info_gain_importance['Info_Gain'])
plt.xticks(range(len(info_gain_importance)), info_gain_importance['Feature'], rotation=45, ha='right')
plt.title('Information Gain Feature Importance')
plt.tight_layout()
plt.show()


# %% [markdown]
# Using the results of these selection algorithms and my own intuition, I ended upselecting the following features and dropping everything else from the datasets.

# %%
selected_features = [
    'sex_male', 
    'age',
    'sibsp',
    'parch',
    'ticket_count', 
    'fare', 
    'pclass',
    'title_Mr',
    'deck_B',
    'deck_G',
    'home.dest_Montreal, PQ',
]

# Now select the features
X_train_selected = X_train_balanced[selected_features]
X_val_selected = X_val[selected_features] 
X_test_selected = X_test[selected_features]


# %% [markdown]
# We had to perform Task 6: Balancing the dataset and Task 7: Feature Selection **after** splitting the data for the following reasons.
# 
# - As already mentioned, we want the validation and test dataset to represent the data distribution of the original dataset because that is what the model will get as input in production. If we balance the data before the split, our validation and test sets also become balanced, which will present the problem of having a distribution shift. Furthermore, our validation and test will contain "fake" synthetic data produced by ADASYN which doesn't really represent real data. For the same reason, our evaluation metrics will be very misleading, even useless.
# - The ADASYN algorithm, used to create synthetic data for oversampling "sees" all the data, including the test and validation data, and uses them to create new records which will be used in the training dataset. This is a big form of data leakage and will make our evaluation metrics misleading.
# - Feature selection with the data before the split will cause our feature selection algorithms to overfit to the data (including validation and test), which would introduce leakage and decrease our model's ability to generalize.

# %% [markdown]
# # Training a Logistic Regression Model

# %% [markdown]
# Finally, after selecting the features, it's time to train the Logistic Regression model. I used Grid Search to perform hyperparameter tuning of Logistic Regression

# %%
# Grid search for logistic regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=SEED, max_iter=1000000), # max_iter is set to 1000000 to avoid convergence warnings
    param_grid,
    cv=5, # 5-fold cross validation
    scoring='accuracy', # optimize for accuracy
    n_jobs=-1, # use all available CPU cores
)

grid_search.fit(X_train_selected, y_train_balanced)
print(f"Best parameters: {grid_search.best_params_}")

# %% [markdown]
# Best parameters: {'C': 1, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'saga'}
# I will now use these parameters to train the model. The 'liblinear' solver proved to give better accuracy for validation though.

# %%
# Initialize and train logistic regression model
lr_model = LogisticRegression(random_state=SEED, C=1, penalty='l2', solver='liblinear', class_weight='balanced', max_iter=1000000)
# C is the inverse of regularization strength. A smaller C means more regularization, which helps prevent overfitting.
# penalty is the type of regularization to use. l1 is less prone to overfitting but can be unstable. l2 is more stable but can underfit.
# solver is the algorithm to use to solve the optimization problem. liblinear is good for small datasets. saga is good for large datasets.

lr_model.fit(X_train_selected, y_train_balanced)

# Make predictions on validation set
val_predictions = lr_model.predict(X_val_selected)

# Calculate accuracy
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")


# %% [markdown]
# The model has an accuracy of 0.8367. This means that it got 83.67% of the predictions correctly. This is not the best, but not bad at all. Especially since we are using a basic linear model like Logistic Regression. 
# 
# Now I will plot the Confusion Matrix and use the `classification_report` package to get a more detailed report on the performance of the model on the validation dataset.

# %%
# Create confusion matrix
cm = confusion_matrix(y_val, val_predictions)

# Plot with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 2. ROC Curve

# Get prediction probabilities
val_probs = lr_model.predict_proba(X_val_selected)[:, 1]
fpr, tpr, _ = roc_curve(y_val, val_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 3. Print Classification Report
print("\nClassification Report:")
print(classification_report(y_val, val_predictions, 
                          target_names=['Did Not Survive', 'Survived']))

# %% [markdown]
# What this means:
# For predicting deaths as positives
# - Out of the predictions of death, 91% were correct. (0.91 precision)
# - Out of all actual deaths, 82% were correctly predicted. (0.82 recall)
# - f1-score of 0.86 means that both precision and recall are fairly high so the model is generally good at identifying positives (deaths).
# 
# Since our validation dataset is imbalanced, having 121 cases that did not survive and 75 that survived, it is better predicting deaths than survival. This is shown in the difference between f1 scores (0.86 vs 0.80)
# 
# Since this is an imbalanced dataset, in terms of deaths vs survival, the Area Under the ROC Curve, is a better measure of discrimination power than accuracy. In this case, it's 89%, as opposed to random 50%. This means that the model performs very well on the validation dataset.

# %%
# Get predictions and probabilities for test set
test_predictions = lr_model.predict(X_test_selected)
test_probs = lr_model.predict_proba(X_test_selected)[:, 1]

# Create confusion matrix
cm_test = confusion_matrix(y_test, test_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title('Test Set Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot ROC curve
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Set ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Print classification report
print("\nTest Set Classification Report:")
print(classification_report(y_test, test_predictions,
                          target_names=['Did Not Survive', 'Survived']))


# %% [markdown]
# Similar analysis to the validation classification report, only that the performance is slightly lower. But the patterns are the same (e.g. the model is better at predicting deaths than survival).
# 
# Still, the AUC-ROC of 0.86 demonstrates that the model performs well.

# %% [markdown]
# # Conclusion
# 
# In this assignment, we performed the intial steps of an AI pipeline: Exploratory Data Analysis, Data Preprocessing, and Feature Engineering.The decent validation and test results of the model show that the steps were followed in good accordance.
# 
# ### EDA
# We first loaded the dataset and printed out the first rows, common statistics of the dataset, and data types. This allowed us to shape a good image of the properties of the data in our head for later manipulation.
# 
# ### Outliers
# We also plotted a boxplot to identify the distribution of outliers. We noticed significant outliers i the `fare` attribute, but all methods of handling them resulted in slightly lower or unchanged model accuracy on the validation dataset, so we ignored them. The methods we used were:
#     - Log transformation
#     - Winsorization
#     - Clipping
#     - Removal
#     - Using Robust Scaler instead of StandardScaler in Scaling stage.
# 
# ### Initial Feature Engineering
# Since the `name` and `ticket` attributes had no missing values, were high-cardinal categorical attributes, and assumed to have low predictive power, we decided to perform initial feature engineering on them since they were only reformatting or counting. The resulting features were:
#     - `title` from `name`
#     - `ticket_count` from `ticket`.
# 
# ### Missing Values before Data Split
# We realized that we wouldn't have to compute statistics from the whole data for imputing the `home.dest` attribute so we did it before splitting the data. We first looked for any shared ticket number with a value to impute, then replaced any remaining empty values with "Unknown".
# 
# ### Splitting the Data
# We used Scikit-learn's `train_test_split` method to split the data. We first split the data into a training (70%) and holdout (30%) set, then split the holdout set in half to obtain a validation (15%) and test (15%) dataset. We used the `stratify=y` argument to keep the same target distribution in the datasets for better training and performance metrics. We then computed some distribution statistics to make sure the datasets had similar distributions. This is important because we want to train on the data we will test the model on.
# 
# ### Missing Values after Data Split
# For `fare` and `embarked`, since there were only 1 and 2 values missing respectively, I manually looked up the missing values, then searched for other people with other similar attributes, and used them to impute those missing values.
# 
# For `age` I grouped the dataset by `pclass` and `sex` and imputed with the median of those groups. I formed this strategy on the training set and performed it seperately on the validation and training datasets to maintain consistency and prevent data leakage.
# 
# For `cabin`, I extracted the `deck` (letter) from the cabin and used KNN to predict the `deck` missing values from the data. I first tried KMeans clustering but KNN proved superior and was more justified. I formed this strategy on the training set and performed it seperately on the validation and training datasets to maintain consistency and prevent data leakage.
# 
# ### Trying Feature Discretization
# In a separate branch, I discretized `age` and `fare` (and separately) to see if there would be an increase in accuracy. There wasn't. So I didn't include the implementations in this notebook.
# 
# ### Encoding
# I first grouped the long tail of the `home.dest` attribute into "Other" category. 
# 
# I decided not to encode `pclass`, `sibsp`, `parch`, and `ticket_count`. This is because the ordinality of these attributes carries some meaning which would be lost in one-hot encoding, and it would add unnecessary sparse dimensions to the features. I left them for the scaling stage to be normalized.
# 
# For the rest of the categorical variables, I used one-hot encoding with the `drop='first'` argument to remove dummy columns that can be inferred from the other categories, and `handle_unknown='ignore'` to ensure categorical consistency while simplifying the encoding process.
# 
# I used the `fit_transform` method on the training dataset and the `transform` method on the validation and test dataset for consistency.
# 
# ### Scaling the Data
# I tried to use the Shapiro-Wilk test to find which numerical features are normal and better served by standardization. The test rejected the normality of all attributes, however, since `age` and `fare` were continuous variables that were somehow close to normal and handling outliers in them was more important, I decided to standardize them and normalize the rest.
# 
# I used the `fit_transform` method on the training dataset and the `transform` method on the validation and test dataset for consistency.
# 
# ### Balancing the data
# I noticed a slight imbalance of the data (62-38) so I decided to employ a data balancing strategy. Since I was already starting with a small dataset, I decided to use an oversampling strategy. Between SMOTE and ADASYN, I chose ADASYN because it takes the density of the data points into account and creates weights to maintain the distribution of the minority class.
# 
# I only applied this on the training set to keep the original distribution for validation and test sets.
# 
# ### Feaeture Selection
# First, I identify the low-variance high-correlation attributes and consider removing them.
# 
# Then, I used Boruta, Lasso, and Mutual Information as automated feature selection algorithms to get an intuition of the imporant features. Finally, I combined them with my own intuition to create the final feature set. This was important to avoid high dimensionality and overfitting.
# 
# ### Training a LogisticRegression model
# Finaly, I trained a LogisticRegression model on my data. I obtained an AUC-ROC of 0.89 on validation and 0.86 on test, which is quite decent.
# 
# 
# 
# 
# 
# 
# 


