#!/usr/bin/env python
# coding: utf-8

# In[182]:


##Importing all necessary library

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
import datetime as dt


# In[183]:


##Loading the dataset


# In[2]:


pd.pandas.set_option('display.max_columns', None)
df=pd.read_csv('Leads.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df=df.replace('Select',np.nan)


# In[7]:


round((df.isnull().sum()/len(df))*100,2)


# In[8]:


df["Asymmetrique Activity Index"].value_counts(normalize=True)


# In[9]:


df=df.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'],axis=1)
    


# In[10]:


round((df.isnull().sum()/len(df))*100,2)


# In[11]:


df['Lead Quality'].value_counts(normalize=True)


# In[12]:


df=df.drop(['Lead Profile','How did you hear about X Education'],axis=1)


# In[13]:


round((df.isnull().sum()/len(df))*100,2)


# In[14]:


df['Lead Quality'].describe()


# In[15]:


sns.countplot(df['Lead Quality'])
plt.show()


# In[16]:


df['Lead Quality']=df['Lead Quality'].replace(np.nan,'Not Sure')


# In[17]:


df['Lead Quality'].value_counts(normalize=True)


# In[18]:


df['City'].value_counts()


# In[19]:


sns.countplot(df['City'])


# In[20]:


df["City"]=df["City"].replace(np.nan,'Mumbai')


# In[21]:


round((df.isnull().sum()/len(df))*100,2)


# In[22]:


df['Tags'].value_counts()


# In[23]:


sns.countplot(df['Tags'])
plt.xticks(rotation=90)
plt.show()


# In[24]:


df['Tags']=df['Tags'].replace(np.nan,'Will revert after reading the email')


# In[25]:


df['What matters most to you in choosing a course'].value_counts()


# In[26]:


sns.countplot(df['What matters most to you in choosing a course'])
plt.xticks(rotation=90)
plt.show()


# In[27]:


df['What matters most to you in choosing a course']=df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[28]:


df['What is your current occupation'].describe()


# In[29]:


sns.countplot(df['What is your current occupation'])
plt.xticks(rotation=90)
plt.show()


# In[30]:


df['What is your current occupation']=df['What is your current occupation'].replace(np.nan,'Unemployed')


# In[31]:


sns.countplot(df['Specialization'])
plt.xticks(rotation=90)
plt.show()


# In[32]:


df['Specialization']=df['Specialization'].replace(np.nan,'Others')


# In[33]:


df['Country'].describe()


# In[34]:


df['Country'].value_counts()


# In[35]:


df['Country']=df['Country'].replace(np.nan,'India')


# In[36]:


round((df.isnull().sum()/len(df))*100,2)


# In[37]:


df.dropna(inplace=True)


# In[38]:


round((df.isnull().sum()/len(df))*100,2)


# # EDA

# In[39]:


##Lead converison


# In[40]:


df['Converted'].value_counts()


# In[41]:


Converted=round(sum(df['Converted'])/len(df['Converted'])*100,2)


# In[42]:


Converted


# In[43]:


## Lead origin
sns.countplot(x='Lead Origin',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[44]:


plt.figure(figsize=(15,8))
sns.countplot(x='Lead Source',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[45]:


df['Lead Source']=df['Lead Source'].replace(['google'],'Google')


# In[46]:


df['Lead Source'].value_counts()


# In[47]:


df['Lead Source']=df['Lead Source'].replace(['Click2call','Social Media','Live Chat','Press_Release','Pay per Click Ads','blog','WeLearn','welearnblog_Home','youtubechannel','testone','NC_EDM'],'Others')


# In[48]:


plt.figure(figsize=(15,8))
sns.countplot(x='Lead Source',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[49]:


sns.boxplot(df.TotalVisits)
plt.show()


# In[50]:


df.TotalVisits.describe(percentiles=[0.05,0.25,0.75,0.9,0.95,0.99])


# In[51]:


df=df[df['TotalVisits']<10.0]


# In[52]:


sns.boxplot(df.TotalVisits)
plt.show()


# In[53]:


sns.boxplot(df['Total Time Spent on Website'])


# In[54]:


df['Total Time Spent on Website'].describe(percentiles=[0.05,0.25,0.75,0.9,0.95,0.99])


# In[55]:


df=df[df['Total Time Spent on Website']<1825.8]


# In[56]:


sns.boxplot(df['Total Time Spent on Website'])


# In[57]:


sns.boxplot(x='Converted',y='Total Time Spent on Website',data=df)
plt.show()


# In[58]:


df['Last Activity'].value_counts()


# In[59]:


plt.figure(figsize=[15,8])
sns.countplot(x='Last Activity',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[60]:


df['Last Activity']=df['Last Activity'].replace(['Had a Phone Conversation','Approached upfront','View in browser link Clicked','Email Received','Visited Booth in Tradeshow','Resubscribed to emails','Email Marked Spam'],'Other Activity')


# In[61]:


plt.figure(figsize=[15,8])
sns.countplot(x='Last Activity',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[62]:


df['Country'].value_counts(ascending=False)


# In[63]:


df['Specialization'].value_counts(ascending=False)


# In[64]:


plt.figure(figsize=[15,8])
sns.countplot(x='Specialization',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[65]:


df['What is your current occupation'].value_counts(ascending=False)


# In[66]:


plt.figure(figsize=[15,8])
sns.countplot(x='What is your current occupation',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[67]:


df.columns


# In[68]:


df['What matters most to you in choosing a course'].value_counts(ascending=False)


# In[69]:


df['Tags'].value_counts(ascending=False)


# In[70]:


df['Tags']=df['Tags'].replace(['Still Thinking','Lost to Others','In confusion whether part time or DLP','Lateral student','Interested in Next batch','Want to take admission but has financial problems','Shall take in the next coming month',
'University not recognized','Recognition issue (DEC approval)'],'Other Tags')


# In[71]:


plt.figure(figsize=[15,8])
sns.countplot(x='Tags',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[72]:


plt.figure(figsize=[15,8])
sns.countplot(x='City',hue='Converted',data=df)
plt.xticks(rotation=90)
plt.show()


# In[73]:


##Dropping irrelevant columns


# In[74]:


df.columns


# In[75]:


df=df.drop(['Lead Number','Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement', 'Through Recommendations','Receive More Updates About Our Courses',
           'Update me on Supply Chain Content', 'Get updates on DM Content','I agree to pay the amount through cheque',
       'A free copy of Mastering The Interview','Country','What matters most to you in choosing a course'],axis=1)


# In[76]:


df.info()


# In[77]:


df.shape


# In[78]:


df.head()


# In[79]:


## Converting Yes/No to 1/0


# In[80]:


df['Do Not Email']=df['Do Not Email'].apply(lambda x: 0 if 'No' else 1)
df['Do Not Call']=df['Do Not Call'].apply(lambda x: 0 if 'No' else 1)


# In[81]:


## Creating dummy variables


# In[82]:


dummy=pd.get_dummies(df[['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity']],drop_first=True)


# In[83]:


dummy.head()


# In[84]:


df=pd.concat([df,dummy],axis=1)


# In[85]:


df.head()


# In[86]:


df=df.drop(['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'],axis=1)


# In[87]:


df.head()


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


X=df.drop(['Prospect ID','Converted'],axis=1)


# In[90]:


y=df['Converted']


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[92]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# In[93]:


##X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


# In[94]:


X_train.head()


# In[95]:


Converted=(sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# In[96]:


import statsmodels.api as sm


# In[97]:


df.info()


# In[ ]:





# In[98]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[99]:


rfe.support_


# In[100]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[101]:


col = X_train.columns[rfe.support_]
col


# In[102]:


X_train.columns[~rfe.support_]


# In[103]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[104]:


col=col.drop(['Tags_invalid number','Tags_number not provided'])


# In[105]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[106]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[107]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[108]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[109]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[110]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[111]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[112]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[114]:


print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[115]:


TP = confusion[1,1] 
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]


# In[116]:


TP / float(TP+FN)


# In[117]:


TN / float(TN+FP)


# In[118]:


print(FP/ float(TN+FP))


# In[119]:


print (TP / float(TP+FP))


# In[120]:


print (TN / float(TN+ FN))


# In[176]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[177]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[178]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[179]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[132]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[133]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[134]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[135]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[136]:


metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[137]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[138]:


TP / float(TP+FN)


# In[139]:


TN / float(TN+FP)


# In[180]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[181]:


TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[143]:


TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[144]:


from sklearn.metrics import precision_score, recall_score


# In[145]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[146]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[147]:


from sklearn.metrics import precision_recall_curve


# In[148]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[149]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[150]:


num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[152]:


X_test = X_test[col]
X_test.head()


# In[153]:


X_test_sm = sm.add_constant(X_test)


# PREDICTIONS ON TEST SET

# In[154]:


y_test_pred = res.predict(X_test_sm)


# In[155]:


y_pred_1 = pd.DataFrame(y_test_pred)


# In[156]:


y_pred_1.head()


# In[157]:


y_test_df = pd.DataFrame(y_test)


# In[158]:


y_test_df['Prospect ID'] = y_test_df.index


# In[159]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[160]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[161]:


y_pred_final.head()


# In[162]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[163]:


y_pred_final.head()


# In[164]:


y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[165]:


y_pred_final.head()


# In[166]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[167]:


y_pred_final.head()


# In[168]:


metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[169]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[170]:


TP = confusion2[1,1]
TN = confusion2[0,0]
FP = confusion2[0,1]
FN = confusion2[1,0]


# In[171]:


TP / float(TP+FN)


# In[172]:


TN / float(TN+FP)


# In[173]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[175]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[ ]:




