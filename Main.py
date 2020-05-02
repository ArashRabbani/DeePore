import DeePore as dp

DataName='DeePore_Compact_Data.h5'
dp.check_get('https://www.1digitalrocksportal.org/projects/215/images/159816/download/',DataName)               

List,MIN,MAX=dp.prep(DataName)

#List=List[0:200]


N=np.int32([0,len(List)*.64,len(List)*.8,len(List)])
TrainList=List[N[0]:N[1]]
EvalList=List[N[1]:N[2]]
TestList=List[N[2]:N[3]]



retrain=1
model=trainmodel(DataName,retrain)  
# check the training performance
plt.plot(model.history[2],model.history[1])
plt.xlabel('Time (s)'); plt.ylabel('Training Loss (MSE)'); plt.rcParams.update({'font.size': 5})

#  Now Testing the Model on the test samples
G=gener(len(TestList),DataName,TestList)
L=next(G)
x=L[0]
y=L[1]
y2=model.predict(L[0])
print('\n# Evaluate on '+ str(TestList.shape[0]) + ' test data')
results=model.evaluate(x,y,batch_size=50)
print('test loss, test acc:', results)
#  Denormalize the predictions
MIN=np.reshape(MIN,(1,y.shape[1]))
MAX=np.reshape(MAX,(1,y.shape[1]))
y=np.multiply(y,(MAX-MIN))+MIN
y2=np.multiply(y2,(MAX-MIN))+MIN
y[:,0]=10**y[:,0]
y2[:,0]=10**y2[:,0]


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