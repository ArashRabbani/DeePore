import DeePore as dp
import glob
path=r"C:\Users\rabar\Dropbox (The University of Manchester)\AR1\BigData\DeePore\DB256NPZ\*.npz";
a=0
for FileName in glob.glob(path):
    A=dp.readsampledata(FileName)
    dp.writeh5slice(A,'Data.h5','X',Shape=[256,256,1])
    a=a+1
    if a>5:
        break
DataName='Data/DeePore_Compact_Data.h5'
dp.check_get('https://www.linktodata',DataName)               
List,MIN,MAX=dp.prep(DataName)
TrainList, EvalList, TestList = dp.splitdata(List)
model=dp.trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0)  
# #  Now Testing the Model on the test samples
# dp.testmodel(model,DataName,TestList,MIN,MAX)
# # dp.predict.absperm('')
# A=dp.readsampledata(FileName='Data/Sample_gray.jpg')
# A=dp.readsampledata(FileName='Data/Sample.jpg')
A=dp.readsampledata(FileName='Data/Sample_large.mat')
B=dp.ecl_distance(A)
dp.show_feature_maps(B)
single_values=dp.predict(model,B,MIN,MAX,res=4.8)
import p,px
px.im(B[0,:,:,1])
# px.im(A[0,:,:,1])