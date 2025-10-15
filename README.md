# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```python
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```
<img width="638" height="447" alt="image" src="https://github.com/user-attachments/assets/ec6097fa-8b44-4b29-8966-54196a39317e" />

```python
df=sns.load_dataset("tips")
df
```
<img width="552" height="404" alt="image" src="https://github.com/user-attachments/assets/1aac046f-a7e7-4398-aee4-06763ff94220" />


```python
sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle='solid',legend="auto")
```

<img width="655" height="461" alt="image" src="https://github.com/user-attachments/assets/315d1339-f833-406c-8570-57e947116d6d" />


```python
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi-:Line Plot')
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
```
<img width="654" height="492" alt="image" src="https://github.com/user-attachments/assets/365fd556-5303-4c66-8b8b-855c8587469e" />


```python
tips=sns.load_dataset('tips')
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tip=tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8,6))
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill')
p2=plt.bar(avg_tip.index,avg_tip,label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
```
<img width="859" height="658" alt="image" src="https://github.com/user-attachments/assets/bd8d62c0-92a2-4fc1-8be5-7606571ce6d3" />


```python
avg_total_bill=tips.groupby('time')['total_bill'].mean()
avg_tip=tips.groupby('time')['tip'].mean()
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill',width=0.4)
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip',width=0.4)
plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Time of Day')
plt.legend()
```
<img width="877" height="540" alt="image" src="https://github.com/user-attachments/assets/db186a95-e0af-4053-acb8-52299624bec4" />


```python
years=range(2000,2012)
apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.9372,0.939]
oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]
plt.bar(years,apples)
plt.bar(years,oranges,bottom=apples)
```
<img width="616" height="445" alt="image" src="https://github.com/user-attachments/assets/5a766e84-c2ce-406f-b3ab-fc9964ccf6fd" />


```python
import seaborn as sns
dt=sns.load_dataset('tips')
sns.barplot(x='day',y='total_bill',hue='sex',data=dt,palette='Set1')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')
```
<img width="689" height="484" alt="image" src="https://github.com/user-attachments/assets/39a8ff6c-20dc-4943-a32b-c662fd950498" />


```python
import pandas as pd
tit=pd.read_csv("/content/titanic_dataset (2).csv")
tit
```
<img width="871" height="750" alt="image" src="https://github.com/user-attachments/assets/220d1fa5-2db0-4dbb-8a7d-be84539a1088" />


```python
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow')
plt.title("Fare of Passenger by Embarked Town")
```
<img width="882" height="594" alt="image" src="https://github.com/user-attachments/assets/23f4d593-8ca2-4602-a1b9-30b99411a3c7" />


```python
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow',hue='Pclass')
plt.title("Fare of Passenger by Embarked Town,Divided by Class")
```
<img width="834" height="501" alt="image" src="https://github.com/user-attachments/assets/f4ed4fc9-310f-47f4-8500-66a7a857cc36" />


```python
tips=sns.load_dataset('tips')
sns.scatterplot(x='total_bill',y='tip',hue='sex',data=tips)
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
```
<img width="697" height="492" alt="image" src="https://github.com/user-attachments/assets/ca29aebf-ab65-4bbc-8b55-91c18ec653db" />


```python
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var=np.random.randn(1000)
num_var=pd.Series(num_var,name="Numerical Variable")
num_var
```
<img width="272" height="446" alt="image" src="https://github.com/user-attachments/assets/fae05551-770c-470e-a238-30de566dac92" />


```python
sns.histplot(data=num_var,kde=True)
```
<img width="688" height="463" alt="image" src="https://github.com/user-attachments/assets/544f70c5-9079-4413-af4d-419dbe742db9" />


```python
sns.histplot(data=tit,x="Pclass",hue="Survived",kde=True)
```
<img width="659" height="455" alt="image" src="https://github.com/user-attachments/assets/bcf3a0d7-3080-47ef-9290-2375c2ef2523" />


```python
import matplotlib.pyplot as plt
np.random.seed(0)
marks=np.random.normal(loc=70,scale=10,size=100)
marks
```
<img width="659" height="367" alt="image" src="https://github.com/user-attachments/assets/5b49ca37-f0ce-4b51-aaec-dd489620bae2" />


```python
sns.histplot(data=marks,bins=10,kde=True,stat='count',cumulative=False,multiple='stack',element='bars',palette='Set1',shrink=0.7)
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of Student Marks')
```
<img width="859" height="500" alt="image" src="https://github.com/user-attachments/assets/76d542c8-845d-4ec3-9a88-2e677b99465a" />



```python
tips=sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue=tips['sex'])
```
<img width="648" height="463" alt="image" src="https://github.com/user-attachments/assets/2d6b5f60-6a9f-42cd-b68d-29d78510577d" />


```python
sns.boxplot(x="day",y="total_bill",hue="smoker",data=tips,linewidth=2,width=0.6,boxprops={"facecolor":"lightblue","edgecolor":"darkblue"},
            whiskerprops={"color":"black","linestyle":"--","linewidth":1.5},capprops={"color":"black","linestyle":"--","linewidth":1.5})
```
<img width="691" height="475" alt="image" src="https://github.com/user-attachments/assets/4840a9ce-dc30-40d0-a303-b3b6e20755b0" />


```python
sns.boxplot(x='Pclass',y='Age',data=tit,palette='rainbow')
plt.title("Age by Passenger Class,Titanic")
```
<img width="830" height="576" alt="image" src="https://github.com/user-attachments/assets/6923083e-b0ed-4a63-9126-cd063fc55c53" />


```python
sns.violinplot(x="day",y="total_bill",hue="smoker",data=tips,linewidth=2,width=0.6,palette="Set3",inner="quartile")
plt.xlabel("Day of the week")
plt.ylabel("Total Bill")
plt.title("Violin Plot of Total Bill by Day and Smoker Status")
```
<img width="676" height="472" alt="image" src="https://github.com/user-attachments/assets/195f82d7-0962-4b77-b061-8eb55a47a306" />


```python
import seaborn as sns
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x='day',y='tip',data=tip)
```
<img width="710" height="476" alt="image" src="https://github.com/user-attachments/assets/6df30572-7f39-4e78-a588-1ed671456e3a" />


```python
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x=tip["total_bill"])
```
<img width="640" height="467" alt="image" src="https://github.com/user-attachments/assets/7e9e3852-14a0-466b-8501-f3eabbba351f" />


```python
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x="tip",y="day",data=tip)
```
<img width="736" height="461" alt="image" src="https://github.com/user-attachments/assets/1e888191-32f4-4f6a-9880-2144ecfb97a6" />


```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3,palette='Set3',alpha=0.8)
```
<img width="730" height="462" alt="image" src="https://github.com/user-attachments/assets/32f8accd-f06c-47ca-8f47-3fba56471247" />


```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='layer',linewidth=3,palette='Set2',alpha=0.8)
```
<img width="701" height="461" alt="image" src="https://github.com/user-attachments/assets/4af05a4a-6ebd-4ed4-ad94-dd1c736eb477" />

```python
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='stack',linewidth=3,palette='Set2',alpha=0.8)
```
<img width="719" height="487" alt="image" src="https://github.com/user-attachments/assets/4667aa45-4157-4aa2-a2fd-aea66b4452ba" />


```python
data=np.random.randint(low=1,high=100,size=(10,10))
print("The data to be plotted:\n")
print(data)
```
<img width="342" height="220" alt="image" src="https://github.com/user-attachments/assets/8871d80c-6975-4e16-9dd1-96e536941993" />


```python
hm=sns.heatmap(data=data)
```
<img width="631" height="423" alt="image" src="https://github.com/user-attachments/assets/efdfdcb3-f971-43e1-94a4-f040b2b126a4" />


```python
tips=sns.load_dataset('tips')
numeric_cols=tips.select_dtypes(include=np.number).columns
corr=tips[numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="plasma",linewidth=0.5)
```
<img width="614" height="450" alt="image" src="https://github.com/user-attachments/assets/5e48e175-0cc7-4919-be51-ea46d4242be0" />


# Result:

 Data Visualization using seaborn python library for the given datas has been performed successfully.
