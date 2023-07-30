#!/usr/bin/env python
# coding: utf-8

# In[18]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('CovidAnalysis').getOrCreate()
from numpy import array
from pyspark.sql.types import IntegerType

from pyspark.ml.regression import LinearRegression


# In[19]:


dataset = spark.read.csv("COVID/StatewiseTestingDetails.csv", inferSchema = True, header = True)


# In[20]:


dataset


# In[21]:


dataset.show()


# In[41]:


dataset = dataset.withColumn("Negative", dataset["Negative"].cast(IntegerType()))
dataset = dataset.dropna(subset = ("Negative", "TotalSamples", "Positive"))


# In[42]:


dataset.show()


# In[43]:


dataset.printSchema()


# In[44]:


dataset.printSchema()


# In[45]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[49]:


vector = VectorAssembler(inputCols = ["TotalSamples", "Negative", "Positive"], outputCol = "Output Features")


# In[50]:


output = vector.transform(dataset)


# In[51]:


output.show()


# In[52]:


output.select("Output Features").show()


# In[53]:


output.columns


# In[81]:


finalized_vector_data = output.select("Date", "State", "Output Features", "Positive")


# In[82]:


finalized_vector_data.show()


# In[83]:


train_data, test_data = finalized_vector_data.randomSplit([0.75, 0.25])


# In[84]:


regressor = LinearRegression(featuresCol="Output Features", labelCol= "Positive")
regressor = regressor.fit(train_data)


# In[85]:


regressor.coefficients


# In[86]:


regressor.intercept


# In[87]:


pred_result = regressor.evaluate(test_data)


# In[91]:


pred_result.predictions.show(40)


# In[ ]:




