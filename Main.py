import DeePore as dp

DataName='DeePore_Compact_Data.h5'

dp.check_get('https://www.1digitalrocksportal.org/projects/215/images/159816/download/',DataName)               

List,MIN,MAX=dp.prep(DataName)

TrainList, EvalList, TestList = dp.splitdata(List)

model=dp.trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0)  

#  Now Testing the Model on the test samples
dp.testmodel(model,DataName,TestList,MIN,MAX)




