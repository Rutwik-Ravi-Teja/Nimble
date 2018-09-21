from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('house').getOrCreate()
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructField,StructType,DoubleType

schem=[StructField('Avg Area Income',DoubleType(),True),StructField('Avg Area House Age',DoubleType(),True),StructField('Avg Area Number of Rooms',DoubleType(),True),StructField('Avg Area Number of Bedrooms',DoubleType(),True),StructField('Area Population',DoubleType(),True),StructField('Price',DoubleType(),True)]
final=StructType(fields=schem)
data=spark.read.csv('USA_House.csv',header=True,schema=final)
assemble=VectorAssembler(inputCols=['Avg Area Income','Avg Area House Age','Avg Area Number of Rooms','Avg Area Number of Bedrooms','Area Population'],outputCol='features')    
output=assemble.transform(data)
final_data=output.select('features','Price')

#splitting data
train_data,test_data=final_data.randomSplit([0.6,0.4])

#training 
lr=LinearRegression(labelCol='Price')
lr_model=lr.fit(train_data)
print('Coefficients of the model:',lr_model.coefficients)
print('intercept value:',lr_model.intercept)

#testing & evaluating
test_results=lr_model.evaluate(test_data)
test_results.residuals.show()
print('RMS_error:',test_results.rootMeanSquaredError)
print('r_squared error:',test_results.r2)

#predicting
unlabelled_data=test_data.select('features')
predictions=lr_model.transform(unlabelled_data)
print('predictions:',predictions.show())




