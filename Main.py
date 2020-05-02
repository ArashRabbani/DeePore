import DeePore as dp

DataName='DeePore_Compact_Data.h5'

dp.check_get('https://www.1digitalrocksportal.org/projects/215/images/159816/download/',DataName)               

List,MIN,MAX=dp.prep(DataName)

TrainList, EvalList, TestList = dp.splitdata(List)

model=dp.trainmodel(DataName,TrainList,EvalList,retrain=0)  

#  Now Testing the Model on the test samples
dp.testmodel(model,DataName,TestList,MIN,MAX)



#  Show prediction of 15 single-value features
import pandas as pd
fig=plt.figure(figsize=(30,40))
plt.rcParams.update({'font.size': 30})
df = pd.read_excel('VarNames.xlsx')
for I in range(15):
    ax = fig.add_subplot(5,3,I+1)
    X=y[:,I]
    Y=y2[:,I]
    plt.scatter(X,Y)
    plt.ylabel('Predicted')
    plt.xlabel('Ground truth')
    plt.tick_params(direction="in")
    plt.text(.5,.9,df.Value[I],horizontalalignment='center',transform=ax.transAxes)
    plt.xlim(np.min(X),np.max(X))
    plt.ylim(np.min(Y),np.max(Y))
    if I==0:
        ax.set_yscale('log')
        ax.set_xscale('log')
plt.savefig('Single-value_Features.png')