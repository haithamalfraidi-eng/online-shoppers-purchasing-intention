import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
CSV="online_shoppers_intention.csv"
df=pd.read_csv(CSV)

# Administrative               int64
# Administrative_Duration    float64
# Informational                int64
# Informational_Duration     float64
# ProductRelated               int64
# ProductRelated_Duration    float64
# BounceRates                float64
# ExitRates                  float64
# PageValues                 float64
# SpecialDay                 float64
# Month                          str
# OperatingSystems             int64
# Browser                      int64
# Region                       int64
# TrafficType                  int64
# VisitorType                    str
# Weekend                       bool
# Revenue                       bool
# duplicated=125
#FalseRev=(((df["Revenue"]==False).sum()/df.shape[0])*100).round(3)
# TrueRev=(((df["Revenue"]==True).sum()/df.shape[0])*100).round(3)

# print("True Reveneu= ",TrueRev,"\n False Revenue= ",FalseRev)

#print(df[["ProductRelated_Duration","Revenue","ProductRelated"]].sort_values(by="ProductRelated_Duration",ascending=False).head(5))

Q3=df["ProductRelated_Duration"].quantile(0.75)
Q1=df["ProductRelated_Duration"].quantile(0.25)

IQR=Q3-Q1
UpperBound=Q3+(1.5*IQR) #=3384.1867837500004


# GroupA =df[(df["ProductRelated_Duration"] > UpperBound) &(df["Revenue"] == True)].shape[0]
# print(GroupA)
# GroupB=df[(df["ProductRelated_Duration"]<=UpperBound)& (df["Revenue"]==True)].shape[0]
# print(GroupB)

#print(df[(df["VisitorType"]=="Returning_Visitor")&(df["Revenue"]==True)].shape)

# print(df.groupby("Revenue")["VisitorType"].value_counts())

#print(df.groupby("VisitorType")["Revenue"].mean()*100)
#print(df.groupby("Month")["Revenue"].value_counts())
#print(df.groupby("Month")["Revenue"].mean()*100)

sns.barplot(df.groupby("VisitorType")["Revenue"].mean()*100)
plt.title("Conversion Rate by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Conversion Rate(%)")
# plt.show()


MonthlyTrafic=df["Month"].value_counts()
sns.barplot(x=MonthlyTrafic.index,y=MonthlyTrafic.values)
plt.xlabel("Month")
plt.ylabel("Total Sessions")
plt.title("Monthly Website Traffic")
# plt.show()

SpecifMonth=df[((df["Month"]=="May") | (df["Month"]=="Nov"))]

# print(SpecifMonth.groupby(["Month","VisitorType"]).size())


# print(SpecifMonth.groupby(["VisitorType","Month"])["Revenue"].mean()*100)

ReturningsData=SpecifMonth[SpecifMonth["VisitorType"]=="Returning_Visitor"]


print(ReturningsData.groupby("Month")["ProductRelated"].mean())

