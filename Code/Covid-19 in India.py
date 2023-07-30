#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('CovidAnalysis').getOrCreate()
from numpy import array
from pyspark.sql.types import IntegerType

from pyspark.ml.regression import LinearRegression


# In[2]:


dataset = spark.read.csv("COVID/covid_19_india.csv", inferSchema = True, header = True)


# In[3]:


dataset


# In[4]:


dataset.show()


# In[5]:


dataset.printSchema()


# In[6]:


dataset = dataset.withColumn("ConfirmedIndianNational", dataset["ConfirmedIndianNational"].cast(IntegerType()))

dataset = dataset.withColumn("ConfirmedForeignNational", dataset["ConfirmedForeignNational"].cast(IntegerType()))

dataset = dataset.dropna(subset = ("ConfirmedIndianNational", "ConfirmedForeignNational", "Cured", "Deaths", "Confirmed"))


# In[7]:


dataset.show()


# In[8]:


dataset.printSchema()


# In[9]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[10]:


vector = VectorAssembler(inputCols = ["ConfirmedIndianNational", "ConfirmedForeignNational", "Cured", "Deaths", "Confirmed"], outputCol = "Output Features")


# In[11]:


output = vector.transform(dataset)


# In[12]:


output.show()


# In[13]:


output.select("Output Features").show()


# In[14]:


output.columns


# In[15]:


finalized_vector_data = output.select("Date", "Time", "State/UnionTerritory", "Output Features", "Confirmed")


# In[16]:


finalized_vector_data.show()


# In[17]:


train_data, test_data = finalized_vector_data.randomSplit([0.75, 0.25])


# In[18]:


regressor = LinearRegression(featuresCol="Output Features", labelCol= "Confirmed")
regressor = regressor.fit(train_data)


# In[19]:


regressor.coefficients


# In[20]:


regressor.intercept


# In[21]:


pred_result = regressor.evaluate(test_data)


# In[23]:


pred_result.predictions.show(40)


# In[ ]:




