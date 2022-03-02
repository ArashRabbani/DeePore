import DeePore as dp

# Comparing different model architectures: 
# 1. check or download the compact data
Data_compact='Data\DeePore_Compact_Data.h5'
dp.check_get('https://zenodo.org/record/4297035/files/DeePore_Compact_Data.h5?download=1',Data_compact) 
# 2. prepare the dataset by removing outliers and creating list of training, evaluation and test samples
List=dp.prep(Data_compact)

# 3. shuffling the dataset
List=dp.shuf(List)

# List=List[1:1000]     #uncomment for a smaller dataset for test purposes
TrainList, EvalList, TestList = dp.splitdata(List)

# 4. defining the training and testing workflows
def calc(I):
    model=dp.trainmodel(Data_compact,TrainList,EvalList,retrain=1,epochs=100,batch_size=100,ModelType=I)  
    dp.testmodel(model,Data_compact,TestList,ModelType=I)

# 5. test different scenarios in parallel
import numpy as np
out=dp.parfor(calc,np.arange(1,10))