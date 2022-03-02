import DeePore as dp
# Retrain and test the model: 
# If you want to try you own architecture of neural network or retrain the present one
# 1. check or download the compact data
Data_compact='Data\DeePore_Compact_Data.h5'
dp.check_get('https://zenodo.org/record/4297035/files/DeePore_Compact_Data.h5?download=1',Data_compact) 

# 2. prepare the dataset by removing outliers and creating list of training, evaluation and test samples
List=dp.prep(Data_compact)
TrainList, EvalList, TestList = dp.splitdata(List)
# 3. retrain the model
model=dp.trainmodel(Data_compact,TrainList,EvalList,retrain=0,epochs=50,batch_size=100,ModelType=3)  
# 4. test the model
dp.testmodel(model,Data_compact,TestList)
