// Databricks notebook source
import org.apache.spark.sql.Encoders

case class employee(Age: Int, Attrition: String,	
                    BusinessTravel:String,	DailyRate: Int, 
                    Department:String,	DistanceFromHome:Int,	
                    Education: Int,	    EducationField:String,	
                    EmployeeCount:Int,	EmployeeNumber:Int, 
                    EnvironmentSatisfaction:Int,	Gender:String,	
                    HourlyRate:Int, 	JobInvolvement:Int, 	
                    JobLevel:Int, 	JobRole:String, 	
                    JobSatisfaction:Int,	MaritalStatus:String, 	
                    MonthlyIncome:Int,	MonthlyRate:Int,	
                    NumCompaniesWorked:Int,	Over18:String,  	
                    OverTime:String,	PercentSalaryHike:Int, 	
                    PerformanceRating:Int,	RelationshipSatisfaction:Int,	
                    StandardHours:Int,	StockOptionLevel:Int,	
                    TotalWorkingYears:Int,	TrainingTimesLastYear:Int,	
                    WorkLifeBalance:Int,	YearsAtCompany:Int,	
                    YearsInCurrentRole:Int,	YearsSinceLastPromotion:Int,	
                    YearsWithCurrManager:Int)

val employeeSchema = Encoders.product[employee].schema
val employeeDF = spark.read.schema(employeeSchema).option("header", "true").csv("/FileStore/tables/EmployeeAttrition.csv")
employeeDF.show()

//import the dataset into the dataframe, read the csv file that we upload to dbfs
//the csv file consists of the header the first row, set header = true
//display the dataframe
employeeDF.count()

//calculate how many records we have in the dataframe, we have 1470 records

// COMMAND ----------

// DBTITLE 1,Summary of our Data
// MAGIC %md
// MAGIC
// MAGIC Summary:
// MAGIC
// MAGIC Dataset Structure: 1470 observations (rows), 35 features (variables)
// MAGIC
// MAGIC Missing Data: Luckily for us, there is no missing data! this will make it easier to work with the dataset. (There is no null value or missing value)
// MAGIC
// MAGIC Data Type: We only have two datatypes in this dataset: factors and integers
// MAGIC
// MAGIC Label" Attrition is the label in our dataset and we would like to find out why employees are leaving the organization!
// MAGIC

// COMMAND ----------

// DBTITLE 1,Finding count, mean, maximum, standard deviation and minimum

employeeDF.select("Age","DailyRate", "DistanceFromHome", "Education", "EmployeeCount", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StandardHours", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager").describe().show()

//select the numeric column in the dataset and display count, mean, maximum,
//standard deviation, minimum, basic stast informations -EDA purposes

// COMMAND ----------

// DBTITLE 1,Printing Schema

employeeDF.printSchema()

//display the schema of the dataframe, each column usee different data types such as string, integer

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 

employeeDF.createOrReplaceTempView("EmployeeData")
//create a tempview object to serve as the table for us to run spark sq queries

// COMMAND ----------

// DBTITLE 1,Querying the Temporary View
// MAGIC %sql
// MAGIC
// MAGIC select * from EmployeeData;
// MAGIC --select all column from the tables

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// DBTITLE 1,Distribution of our Labels:
// MAGIC %md
// MAGIC
// MAGIC This is an important aspect that will be further discussed is dealing with imbalanced dataset. 84% of employees did not quit the organization while 16% did leave the organization. Knowing that we are dealing with an imbalanced dataset will help us determine what will be the best approach to implement our predictive model.

// COMMAND ----------

// DBTITLE 1,Displaying the Number(count) of Employee Attrition (Yes/No)
// MAGIC %sql
// MAGIC
// MAGIC select Attrition as Employee_Attrition, count(Attrition) as counts  from EmployeeData group by Attrition;
// MAGIC -- how many staff churn or stays? why

// COMMAND ----------

// DBTITLE 1,Displaying Percentage of Employee Leaving the Organization
// MAGIC %sql
// MAGIC
// MAGIC select count(Attrition), Attrition from EmployeeData group by Attrition;
// MAGIC -- how many staff churn or stays, display in pie chart with %, why?

// COMMAND ----------

// DBTITLE 1,Gender Analysis
// MAGIC %md
// MAGIC
// MAGIC In this section, we will try to see if there are any discrepancies between male and females in the organization. Also, we will look at other basic information such as the age, level of job satisfaction and average salary by gender. 
// MAGIC
// MAGIC
// MAGIC #### Questions to ask Ourselves:
// MAGIC
// MAGIC * What is the age distribution between males and females? Are there any significant discrepancies?.
// MAGIC * What is the average job satisfaction by attrition status? Is any type of gender more disatisfied than the other?
// MAGIC * What is the average salary by gender? What are the number of employees by Gender in each department?
// MAGIC
// MAGIC
// MAGIC #### Summary:
// MAGIC
// MAGIC * Age by Gender: The average age of females is 37.33 and for males is 36.65 and both distributions are similar.
// MAGIC * Job Satisfaction by Gender: For individuals who didn't leave the organization, job satisfaction levels are practically the same. However, for people who left the organization, females had a lower satisfaction level as opposed to males.
// MAGIC * Salaries: The average salaries for both genders are practically the same with males having an average of 6380.51 and females 6686.57
// MAGIC * Departments: There are a higher number of males in the three departments however, females are more predominant in the Research and Development department.

// COMMAND ----------

// DBTITLE 1,Age Distribution by Gender
// MAGIC %sql
// MAGIC
// MAGIC select mean(Age), Gender from EmployeeData group by Gender ;
// MAGIC -- how many male and female staff we have and whats is their avg age
// MAGIC -- is there any discrimination in terms of gender
// MAGIC -- we do not discriminate by gender

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select mean(Age) from EmployeeData ;
// MAGIC -- what is the avg age for all staff, all staff on avg they are 36 to 37 years old

// COMMAND ----------

// DBTITLE 1,Distribution of Job Satisfaction:
// MAGIC %sql
// MAGIC
// MAGIC select JobSatisfaction, Attrition from EmployeeData;
// MAGIC -- does job satisfaction correlated to churn rate? yes or no and why?
// MAGIC -- does job satisfaction affect the churn rate? why?
// MAGIC -- those who churn (yes) their job satisfaction is range from 1 to 3
// MAGIC -- those who stays (no) their job satisfaction is more than 2
// MAGIC -- yes, job satisfaction can affect the churn rate

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(JobSatisfaction), JobSatisfaction from EmployeeData group by JobSatisfaction;
// MAGIC
// MAGIC --what is the distribution of job satisfaction score (histogram) for all employees

// COMMAND ----------

// DBTITLE 1,Monthly Income by Gender
// MAGIC %sql
// MAGIC
// MAGIC select Gender, MonthlyIncome from EmployeeData;
// MAGIC -- does male or female employee earn more, does the organization practice pay discrimination by genders
// MAGIC -- is there any significant in terms of pay due to the gender differences

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC # Average Income and Presence by Department
// MAGIC

// COMMAND ----------

// DBTITLE 1,Average Salary by Gender
// MAGIC %sql
// MAGIC
// MAGIC select avg(MonthlyIncome), Gender from EmployeeData group by Gender;
// MAGIC -- calculate the avg income for both genders to help us to determine is there any pay differences due to gender differences 
// MAGIC -- the previous cell above does not give us a clear picture yet

// COMMAND ----------

// DBTITLE 1,Number of Employees by Department
// MAGIC %sql
// MAGIC
// MAGIC select count(Department) as Number_of_Employee, Department, Gender from EmployeeData group by Department, Gender;
// MAGIC -- how many male and female staff works in different departments
// MAGIC -- does male or female staff dominates specific departments
// MAGIC -- any department dominated by specific gender for example in R&D department, significant gaps

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(Department) as Number_of_Employee, Department from EmployeeData group by Department;
// MAGIC -- how many employee works in each departments, R&D is the largest department with the largest number of staffs
// MAGIC -- it accounts of 65% of the total staff, imagine if high turn over rate happens in this department

// COMMAND ----------

// DBTITLE 1,Analysis by Generation and Education:
// MAGIC %md
// MAGIC It is well known, that each type of generation have their particular peculiarities and that is something I decided we should explore in this dataset. Nevertheless, there is still more coming in this section and I wonder what differences does each generation have when it comes to this dataset. 
// MAGIC
// MAGIC   
// MAGIC #### Questions to Ask Ourselves:
// MAGIC
// MAGIC * What is the average number of companies previously worked for each generation? Our aim is to see if it is true that past generations used to stay longer in one company.
// MAGIC
// MAGIC #### Summary:
// MAGIC
// MAGIC * Employees who quit the organization: For these type of employees we see that the boomers had a higher number of companies previously worked at.
// MAGIC * Millenials: Most millenials are still relatively young, so that explains why the number of companies for millenials is relatively low however, I expect this number to increase as the years pass by.
// MAGIC * Attrition by Generation: It seems that millenials are the ones with the highest turnover rate, followed by the boomers. What does this tell us? The newer generation which are the millenials opt to look more easy for other jobs that satisfy the needs on the other side we have the boomers which are approximating retirement and could be one of the reasons why the turnover rate of boomers is the second highest.
// MAGIC * Attrition by Level of Education: This goes hand in hand with the previous statement, as bachelors are the ones showing the highest level of attrition which makes sense since Millenials create the highest turnover rate inside the organization.

// COMMAND ----------

// DBTITLE 1,Understanding Generational Behavior:
// MAGIC %sql
// MAGIC
// MAGIC SELECT NumCompaniesWorked,
// MAGIC CASE
// MAGIC     WHEN Age < 37 THEN "Millenials"
// MAGIC     WHEN Age >= 38 AND Age < 54 THEN "Generation X"
// MAGIC     WHEN Age >= 54 AND Age < 73 THEN "Boomers"
// MAGIC     ELSE "Silent"
// MAGIC END AS Generation
// MAGIC FROM EmployeeData;
// MAGIC
// MAGIC -- determine the generation based on the employee age
// MAGIC -- does it mean younger generations like to switch jobs compared with older generation

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC SELECT NumCompaniesWorked,Attrition,
// MAGIC CASE
// MAGIC     WHEN Age < 37 THEN "Millenials"
// MAGIC     WHEN Age >= 38 AND Age < 54 THEN "Generation X"
// MAGIC     WHEN Age >= 54 AND Age < 73 THEN "Boomers"
// MAGIC     ELSE "Silent"
// MAGIC END AS Generation
// MAGIC FROM EmployeeData;
// MAGIC
// MAGIC -- determine the correlation between generation and churn rate
// MAGIC -- which generation churn more, why?

// COMMAND ----------

// DBTITLE 1,Attrition by Educational Level:
// MAGIC %sql
// MAGIC
// MAGIC SELECT Attrition,count(Education),
// MAGIC CASE
// MAGIC     WHEN Education == 1 THEN "Without College Degree"
// MAGIC     WHEN Education == 2 THEN "College Degree"
// MAGIC     WHEN Education == 3 THEN "Bachelors Degree"
// MAGIC     WHEN Education == 4 THEN "Masters Degree"
// MAGIC     ELSE "Phd Degree"
// MAGIC END AS Education
// MAGIC FROM EmployeeData
// MAGIC Group by Education,Attrition;
// MAGIC
// MAGIC -- does education level affect the churn rate? why?
// MAGIC -- which gorup churn more? why?

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC SELECT Attrition,
// MAGIC CASE
// MAGIC     WHEN Education == 1 THEN "Without College Degree"
// MAGIC     WHEN Education == 2 THEN "College Degree"
// MAGIC     WHEN Education == 3 THEN "Bachelors Degree"
// MAGIC     WHEN Education == 4 THEN "Masters Degree"
// MAGIC     ELSE "Phd Degree"
// MAGIC END AS Education
// MAGIC FROM EmployeeData;
// MAGIC
// MAGIC -- most of the staff who churn are bachelor degree holder, follwoed by mastere degree, college degree
// MAGIC -- compare the staff who churns and stays, education lavel, does it correlate the same way

// COMMAND ----------

// DBTITLE 1,The Impact of Income towards Attrition
// MAGIC %md
// MAGIC
// MAGIC I wonder how much importance does each employee give to the income they earn in the organization. Here we will find out if it is true that money is really everything!

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #### Questions to Ask Ourselves
// MAGIC
// MAGIC * What is the average monthly income by department? Are there any significant differences between individuals who quit and didn't quit?
// MAGIC * Are there significant changes in the level of income by Job Satisfaction? Are individuals with a lower satisfaction getting much less income than the ones who are more satisfied?
// MAGIC * Do employees who quit the organization have a much lower income than people who didn't quit the organization?
// MAGIC * Do employees with a higher performance rating earn more than with a lower performance rating? Is the difference significant by Attrition status?
// MAGIC
// MAGIC #### Summary:
// MAGIC
// MAGIC * Income by Departments: Wow! We can see huge differences in each department by attrition status.
// MAGIC * Income by Job Satisfaction: Hmm. It seems the lower the job satisfaction the wider the gap by attrition status in the levels of income.
// MAGIC * Attrition sample population: I would say that most of this sample population has had a salary increase of less than 15% and a monthly income of less than 7,000
// MAGIC * Exhaustion at Work: Over 54% of workers who left the organization worked overtime! Will this be a reason why employees are leaving?
// MAGIC * Differences in the DailyRate: HealthCare Representatives , Sales Representatives , and Research Scientists have the biggest daily rates differences in terms of employees who quit or didn't quit the organization. This might indicate that at least for the these roles, the sample population that left the organization was mainly because of income.

// COMMAND ----------

// DBTITLE 1,Average Income by Department:
// MAGIC %sql
// MAGIC
// MAGIC select Attrition,  Department, avg(MonthlyIncome) from EmployeeData group by Attrition,  Department;
// MAGIC
// MAGIC --does the department and monthly income affect the churn rate or not? why?
// MAGIC -- what is the correlation between dept, monthly income with churn rate?
// MAGIC -- sales dept, r&d dept has a higher churn rate, the staff who churn have a lowe salary.

// COMMAND ----------

// DBTITLE 1,Determining Satisfaction by Income:
// MAGIC %sql
// MAGIC
// MAGIC select Attrition,  JobSatisfaction, cast(mean(MonthlyIncome) as int) as Median_Income from EmployeeData group by Attrition,  JobSatisfaction;
// MAGIC
// MAGIC -- does the job satisfaction and monthly income affect the churn rate? why?
// MAGIC -- what is the correlation between job satisfaction, income and churn rate
// MAGIC -- compare the pie chart below, job satisfaction not really affect the churn rate
// MAGIC -- seems monthly income affect the churn rate, poeple who stays earns more 6k plus with 40% job satisfaction rate
// MAGIC -- people with the same job satisfaction rate churns because of lower pay, 4k plus only

// COMMAND ----------

// DBTITLE 1,Average and Percent Difference of Daily Rates:
// MAGIC %sql
// MAGIC
// MAGIC select JobRole, DailyRate, Attrition from EmployeeData group by JobRole, DailyRate, Attrition; 
// MAGIC -- what is the correlation between job role, daily rate and churn rate? does it affect each other? why?
// MAGIC -- yes, some job roles has a higher churn rate such as research scientist, sales rep, lab technician, sales executive
// MAGIC -- the staff who churns earns less than those staff who stay, thus the daily rate can affect the churn rate

// COMMAND ----------

// DBTITLE 1,Level of Attrition by Overtime Status:
// MAGIC %sql
// MAGIC
// MAGIC select  OverTime,count(OverTime), Attrition from EmployeeData group by OverTime, Attrition; 
// MAGIC -- does the overtime requirement affect the churn rate or not?
// MAGIC -- work life balance, does this affect churn rate? why?
// MAGIC -- 54% of the staff who churn are required to work overtime compared with the staff who stays 77% no need to work overtime
// MAGIC -- the job role affect the overtime needs, some job role need to work overtime

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC # Working Environment
// MAGIC
// MAGIC In this section, we will explore everything that is related to the working environment and the structure of the organization.
// MAGIC
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #### Questions to ask Ourselves
// MAGIC
// MAGIC * Job Roles: How many employees in each Job Role?
// MAGIC * Salary by Job Role: What's the average salary by job role?
// MAGIC * Attrition by Job Role: What's the attrition percentage by job role? Which job role has the highest attrition rate? Which has the lowest?
// MAGIC * Years with Current Manager What's the average satisfaction rate by the status of the manager? Are recently hired managers providinga higher job satisfaction to employees?
// MAGIC * Working Environment by Job Role: What's the working environment by job role?
// MAGIC
// MAGIC #### Summary:
// MAGIC
// MAGIC * Number of Employees by Job Role: Sales and Research Scientist are the job positions with the highest number of employees.
// MAGIC * Salary by Job Role: Managers and Research Directors have the highest salary on average.
// MAGIC * Attrition by Job Role: Sales Representatives, HealthCare Representatives and Managers have the highest attrition rates. This could give us a hint that in these departments we are experiencing certain issues with employees.
// MAGIC * Managers: Employees that are dealing with recently hired managers have a lower satisfaction score than managers that have been there for a longer time.
// MAGIC * Working Environment: As expected, managers and healthcare representatives are dealing with a lower working environment however, we don't see the same with sales representatives that could be because most sales representatives work outside the organization.

// COMMAND ----------

// DBTITLE 1,Median Salary by Job Role
// MAGIC %sql
// MAGIC
// MAGIC select mean(MonthlyIncome) as Median_Salary , JobRole, Attrition from  EmployeeData group by JobRole, Attrition order by Median_Salary desc;
// MAGIC
// MAGIC -- does the income, job role affect the churn rate? why?

// COMMAND ----------

// DBTITLE 1,Attrition by Job Role
// MAGIC %sql
// MAGIC
// MAGIC select JobRole, Attrition, count(Attrition) from  EmployeeData group by JobRole, Attrition;
// MAGIC
// MAGIC

// COMMAND ----------

// DBTITLE 1,Current Managers and Average Satisfaction Score:
// MAGIC %sql
// MAGIC
// MAGIC select YearsWithCurrManager, avg(JobSatisfaction), Attrition from EmployeeData group by YearsWithCurrManager, Attrition order by YearsWithCurrManager asc;
// MAGIC
// MAGIC -- does the duration with the current manager, relationship and job satisfaction affect the churn rate? why?

// COMMAND ----------

// DBTITLE 1,Average Environment Satisfaction:
// MAGIC %sql
// MAGIC
// MAGIC select Attrition, avg(JobSatisfaction), JobRole from EmployeeData group by JobRole, Attrition;
// MAGIC

// COMMAND ----------

// DBTITLE 1,An In-Depth Look into Attrition:
// MAGIC %md
// MAGIC
// MAGIC Digging into Attrition:
// MAGIC
// MAGIC In this section, we will go as deep as we can into employees that quit to have a better understanding what were some of the reasons that employees decided to leave the organization.
// MAGIC
// MAGIC Questions to Ask Ourselves:
// MAGIC
// MAGIC Attrition by Department: How many employees quit by Department? Did they have a proper work-life balance?
// MAGIC Distance from Work: Is distance from work a huge factor in terms of quitting the organization?

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select count(Department) as Number_of_Employees, Department,WorkLifeBalance from EmployeeData group by Department,WorkLifeBalance order by Number_of_Employees desc;
// MAGIC
// MAGIC -- we want to get more details on the number of employees in a dept, dept name, work life balance details
// MAGIC -- does these 2 factors affect the churn rate
// MAGIC -- R&D dept is the largest dept in the organization with the largest number of staff
// MAGIC -- followed by sales dept 
// MAGIC

// COMMAND ----------

// DBTITLE 1,Other Factors that could Influence Attrition:
// MAGIC %md
// MAGIC
// MAGIC In this section we will analyze other external factors that could have a possible influence on individuals leaving the organization. 
// MAGIC Some of the factors include:
// MAGIC
// MAGIC * Home Distance from Work
// MAGIC * Business Travel
// MAGIC * Stock Option Levels
// MAGIC

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select percentile_approx(DistanceFromHome, 0.5) as Median from EmployeeData;

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select Attrition, 
// MAGIC CASE
// MAGIC     WHEN DistanceFromHome < 7 THEN "Below Average"
// MAGIC     ELSE "Above Average"
// MAGIC END AS DistanceFromHome 
// MAGIC from EmployeeData group by DistanceFromHome, Attrition;
// MAGIC
// MAGIC -- does the working distance affect the churn rate? why?
// MAGIC -- yes, more staff will churn because of the longer working distance from home, need more time to travel
// MAGIC -- travelling time to work place can affect the churn rate

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select StockOptionLevel, count(StockOptionLevel), Attrition, count(Attrition) from EmployeeData group by StockOptionLevel, Attrition order by StockOptionLevel desc;
// MAGIC
// MAGIC -- does the stock option plan affect the churn rate? why?

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select BusinessTravel, count(Attrition) as Number_of_Employees, Attrition from EmployeeData group by  BusinessTravel, Attrition
// MAGIC
// MAGIC -- does the travel requirement affect the churn rate? why? not really, could be other factors also 

// COMMAND ----------

// MAGIC %md ## Creating a Classification Model
// MAGIC
// MAGIC In this Project, you will implement a classification model **(Decision tree classifier)** that uses features of employee details and we will predict it is Attrition (Yes or No)
// MAGIC
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC
// MAGIC First, import the libraries you will need:

// COMMAND ----------


import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this project, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **Attrition** column to **label**.

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC
// MAGIC VectorAssembler():  is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. 
// MAGIC
// MAGIC **VectorAssembler** accepts the following input column types: **all numeric types, boolean type, and vector type.** 
// MAGIC
// MAGIC In each row, the **values of the input columns will be concatenated into a vector** in the specified order.

// COMMAND ----------


var StringfeatureCol = Array("Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime")

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ###StringIndexer
// MAGIC
// MAGIC StringIndexer encodes a string column of labels to a column of label indices.

// COMMAND ----------

// DBTITLE 1,Example of StringIndexer
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

indexed.show()

// COMMAND ----------

// MAGIC %md ### Define the Pipeline
// MAGIC A predictive model often requires multiple stages of feature preparation. 
// MAGIC
// MAGIC A pipeline consists of a series of *transformer* and *estimator* stages that typically prepare a DataFrame for modeling and then train a predictive model. 
// MAGIC
// MAGIC In this case, you will create a pipeline with stages:
// MAGIC
// MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
// MAGIC - A **VectorAssembler** that combines categorical features into a single vector

// COMMAND ----------


import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

val indexers = StringfeatureCol.map { colName =>
  new StringIndexer().setInputCol(colName).setHandleInvalid("skip").setOutputCol(colName + "_indexed")
}

val pipeline = new Pipeline()
                    .setStages(indexers)      

val EmpDF = pipeline.fit(employeeDF).transform(employeeDF)

// COMMAND ----------


EmpDF.printSchema()

// COMMAND ----------


EmpDF.show()

// COMMAND ----------


EmpDF.count()

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, you will use 70% of the data for training, and reserve 30% for testing. 

// COMMAND ----------


val splits = EmpDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------


import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("Age",	"BusinessTravel_indexed",	"DailyRate",	"Department_indexed",	"DistanceFromHome",	"Education",	"EducationField_indexed",	"EnvironmentSatisfaction",	"Gender_indexed",	"HourlyRate",	"JobInvolvement",	"JobLevel",	"JobRole_indexed",	"JobSatisfaction",	"MaritalStatus_indexed",	"MonthlyIncome",	"MonthlyRate",	"NumCompaniesWorked",	"Over18_indexed",	"OverTime_indexed",	"PercentSalaryHike",	"PerformanceRating",	"RelationshipSatisfaction",	"StandardHours",	"StockOptionLevel",	"TotalWorkingYears",	"TrainingTimesLastYear",	"WorkLifeBalance",	"YearsAtCompany",	"YearsInCurrentRole",	"YearsSinceLastPromotion",	"YearsWithCurrManager")).setOutputCol("features")

val training = assembler.transform(train).select($"features", $"Attrition_indexed".alias("label"))

training.show()

// COMMAND ----------

// MAGIC %md ### Train a Classification Model (Decision tree classifier)
// MAGIC Next, you need to train a Classification Model using the training data. To do this, create an instance of the Decision tree classifier algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Decision tree classifier* algorithm 

// COMMAND ----------


import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val lr = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

val model = lr.fit(training)

println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **Churn_indexed** column to **trueLabel**.

// COMMAND ----------


val testing = assembler.transform(test).select($"features", $"Attrition_indexed".alias("trueLabel"))
testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Attrition. 

// COMMAND ----------


val prediction = model.transform(testing)
val predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100)

// COMMAND ----------

// MAGIC %md Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*) you'll learn how to measure the accuracy of a model.

// COMMAND ----------

// MAGIC %md ### Classification model Evalation
// MAGIC
// MAGIC spark.mllib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. spark.mllib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

// COMMAND ----------


val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("trueLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #Top Reasons why Employees leave the Organization:
// MAGIC
// MAGIC * No Overtime This was a surpirse, employees who don't have overtime are most likely to leave the organization. This could be that employees would like to have a higher amount of income or employees could feel that they are underused.
// MAGIC * Monthly Income: As expected, Income is a huge factor as why employees leave the organization in search for a better salary.
// MAGIC * Age: This could also be expected, since people who are aiming to retire will leave the organization.
// MAGIC
// MAGIC Knowing the most likely reasons why employees leave the organization, can help the organization take action and reduce the level of Attrition inside the organization.

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC # An Interesting Quote I Found:
// MAGIC #### " Managers tend to blame their turnover problems on everything under the sun, while ignoring the crux of the matter: people don't leave jobs; they leave managers." by Travis BradBerry 

// COMMAND ----------

//try use logistics regression model to do so
// and compare the accuracy of both model
//homework

//you can try use logistic regression model

//to predict

//measure the accuracy with confusion matrix
