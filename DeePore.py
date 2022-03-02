import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
import os, sys
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
import scipy.io as sio
from scipy.ndimage import distance_transform_edt as distance
import tensorflow.keras.backend as kb   
# import cv2

def check_get(url,File_Name): 
    def download_callback(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d MB / %d MB" % (
                percent, len(str(totalsize)), readsofar/1e6, totalsize/1e6)
            sys.stderr.write(s)
            if readsofar >= totalsize: # near the end
                sys.stderr.write("\n")
        else: # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))
    if not os.path.isfile(File_Name):
        ans=input('You dont have the file "' +File_Name +'". Do you want to download it? (Y/N) ')    
        if ans=='Y' or ans=='y' or ans=='yes' or ans=='Yes' or ans=='YES':
            print('Beginning file download. This might take several minutes.')
            urlretrieve(url,File_Name,download_callback)
    else:
        print('File "' +File_Name +'" is detected on your machine.'  )
def shuf(L):
    import random
    random.shuffle(L)
    return L
def WMSE(y_actual,y_pred): # weighted MSE loss
    w=np.ones((1,1515))
    w[:,15:]=.01
    w=np.float32(w)
    w=tf.tile(w, [tf.shape(y_pred)[0],1])
    loss=kb.square(y_actual-y_pred)
    loss=tf.multiply(loss,w)
    return loss
def WBCE(y_actual,y_pred): # weighted binary crossentropy loss
    w=np.ones((1,1515))
    w[:,15:]=.01
    w=np.float32(w)
    w=tf.tile(w, [tf.shape(y_pred)[0],1])
    loss=kb.square(y_actual-y_pred)
    loss=y_actual*(-tf.math.log(y_pred))+(1-y_actual)*(-tf.math.log(1-y_pred))
    loss=tf.multiply(loss,w)
    return loss

def DeePore1(INPUT_SHAPE,OUTPUT_SHAPE):
    # variable filters / 3 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse']) 
    return model
def DeePore2(INPUT_SHAPE,OUTPUT_SHAPE):
    # variable filters / 4 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)    
    c4 = Conv2D(24, (2, 2), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4)      
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse']) 
    return model
def DeePore3(INPUT_SHAPE,OUTPUT_SHAPE):
    # fixed filter size/ 3 convs
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss='mse', metrics=['mse']) 
    return model

def DeePore4(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse']) 
    return model
def DeePore5(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)    
    c4 = Conv2D(24, (2, 2), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4)      
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse']) 
    return model
def DeePore6(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    optim=tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer=optim, loss=WMSE, metrics=['mse']) 
    return model
def DeePore7(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (8, 8), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (4, 4), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (2, 2), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse']) 
    return model
def DeePore8(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)     
    f=tf.keras.layers.Flatten()(p3)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse']) 
    return model
def DeePore9(INPUT_SHAPE,OUTPUT_SHAPE):
    inputs = Input(INPUT_SHAPE[1:])
    c1 = Conv2D(6, (3, 3), kernel_initializer='he_normal', padding='same') (inputs)
    p1 = MaxPooling2D((2, 2)) (c1)    
    c2 = Conv2D(12, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    p2 = MaxPooling2D((2, 2)) (c2) 
    c3 = Conv2D(18, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    p3 = MaxPooling2D((2, 2)) (c3)   
    c4 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same') (p3)
    p4 = MaxPooling2D((2, 2)) (c4) 
    f=tf.keras.layers.Flatten()(p4)
    d1=tf.keras.layers.Dense(1515, activation=tf.nn.leaky_relu)(f)
    d2=tf.keras.layers.Dense(1515, activation=tf.nn.sigmoid)(d1)
    outputs=d2
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=WBCE, metrics=['mse']) 
    return model

def prep(Data):
    print('Cheching the data for outliers. Please wait...')
    List=[]
    with h5py.File(Data,'r') as f:
        length=f['X'].shape[0]
        MIN=np.ones((f['Y'].shape[1],1))*1e7
        MAX=-MIN
        counter=0
        for I in range(length):
            t2=f['Y'][counter,...]
            y=t2.astype('float32')
            D=int(np.sum(np.isnan(y)))+int(np.sum(np.isinf(y)))+int(y[1]>120)+int(y[4]>1.8)+int(y[0]<1e-4)+int(y[2]<1e-5)+int(y[14]>.7)

            if D>0:
                pass
            else:
                List=np.append(List,counter)
                y[0:15]=np.log10(y[0:15]) # applying log10 to handle range of order of magnitudes
                maxid=np.argwhere(y>MAX)
                minid=np.argwhere(y<MIN)
                MAX[maxid[:,0]]=y[maxid[:,0]]
                MIN[minid[:,0]]=y[minid[:,0]]
            if counter % 100==0:
                print('checking sample'+str(counter))
            counter=counter+1
        Singles=15
        for I in range(15):
            MAX[Singles+100*I:Singles+100*(I+1)]=np.max(MAX[Singles+100*I:Singles+100*(I+1)])
            MIN[Singles+100*I:Singles+100*(I+1)]=np.min(MIN[Singles+100*I:Singles+100*(I+1)])
    np.save('minmax.npy',[MIN,MAX])
    return List

def gener(batch_size,Data,List,MIN,MAX):
    with h5py.File(Data,'r') as f:
        length=len(List)
        samples_per_epoch = length
        number_of_batches = int(samples_per_epoch/batch_size)
        counter=0
        while 1:
            t1=f['X'][np.int32(np.sort(List[batch_size*counter:batch_size*(counter+1)])),...]
            t2=f['Y'][np.int32(np.sort(List[batch_size*counter:batch_size*(counter+1)])),...]
            X_batch=t1.astype('float32')/128
            y_batch=t2.astype('float32')
            y_batch=np.reshape(y_batch,(y_batch.shape[0],y_batch.shape[1]))
            y_batch[:,0:15]=np.log10(y_batch[:,0:15])
            Min=np.tile(np.transpose(MIN),(batch_size,1))
            Max=np.tile(np.transpose(MAX),(batch_size,1))
            y_batch=(y_batch-Min)/(Max-Min)
            counter += 1
            ids=shuf(np.arange(np.shape(y_batch)[0]))
            X_batch=X_batch[ids,...]
            y_batch=y_batch[ids,...]
            # print(ids)
            yield X_batch,y_batch
            if counter >= number_of_batches: #restart counter to yeild data in the next epoch as well
                counter = 0
def hdf_shapes(Name,Fields):
    # Fields is list of hdf file fields
    Shape = [[] for _ in range(len(Fields))]
    with h5py.File(Name, 'r') as f:
        for I in range(len(Fields)):
            Shape[I]=f[Fields[I]].shape  
    return Shape  
def now():
    import datetime
    d1 = datetime.datetime(1, 1, 1)
    d2 = datetime.datetime.now()
    d=d2-d1
    dd=d.days+d.seconds/(24*60*60)+d.microseconds/(24*60*60*1e6)+367
    return dd      
def nowstr():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d-%b-%Y %H.%M.%S")       
def modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType):
    if ModelType==1:
        model=DeePore1(INPUT_SHAPE,OUTPUT_SHAPE)
    if ModelType==2:
        model=DeePore2(INPUT_SHAPE,OUTPUT_SHAPE)
    if ModelType==3:
        model=DeePore3(INPUT_SHAPE,OUTPUT_SHAPE)    
    if ModelType==4:
        model=DeePore4(INPUT_SHAPE,OUTPUT_SHAPE)  
    if ModelType==5:
        model=DeePore5(INPUT_SHAPE,OUTPUT_SHAPE) 
    if ModelType==6:
        model=DeePore6(INPUT_SHAPE,OUTPUT_SHAPE) 
    if ModelType==7:
        model=DeePore7(INPUT_SHAPE,OUTPUT_SHAPE) 
    if ModelType==8:
        model=DeePore8(INPUT_SHAPE,OUTPUT_SHAPE)    
    if ModelType==9:
        model=DeePore9(INPUT_SHAPE,OUTPUT_SHAPE)  
    return model
 
def trainmodel(DataName,TrainList,EvalList,retrain=0,reload=0,epochs=100,batch_size=100,ModelType=9):
    from tensorflow.keras.callbacks import ModelCheckpoint
    MIN,MAX=np.load('minmax.npy')    
    SaveName='Model'+str(ModelType)+'.h5';
    INPUT_SHAPE,OUTPUT_SHAPE =hdf_shapes(DataName,('X','Y')); 
    OUTPUT_SHAPE=[1,1]
    # callbacks
    timestr=nowstr()
    LogName='log_'+timestr+'_'+'Model'+str(ModelType)
    filepath=SaveName
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_freq=50, save_best_only=True, mode='min')
    with open("Logs/"+LogName+".txt", "wt") as f:
        f.write('# Path to train file: \n')
        f.write(DataName +'\n')
        f.write('# Start time: \n')
        f.write(timestr +'\n')
        nowstr()
        st='# Training loss'
        spa=' ' * (40-len(st))
        st=st+spa+'Validation loss'
        f.write(st+'\n')

    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.val_loss_=None
            self.start_time=now()
        def on_batch_end(self, batch, logs=None):
            if self.val_loss_==None:
                self.val_loss_=logs['mse']
            with open("Logs/"+LogName+".txt", "a") as f:
                st=str(logs['mse'])
                spa=' ' * (40-len(st))
                st=st+spa+str(self.val_loss_)
                f.write(st+'\n')
        def on_test_batch_end(self,batch, logs=None):
            self.val_loss_=logs['mse']
    callbacks_list = [checkpoint,MyCallback()]
    callbacks_list = [checkpoint]
    callbacks_list = []
    callbacks_list = [MyCallback()]
    # end of callbacks
    model=modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType)
    
 
    if retrain:
        if reload:
            model.load_weights(SaveName)            
        
        model.fit(gener(batch_size,DataName,TrainList,MIN,MAX), epochs=epochs,steps_per_epoch=int(len(TrainList)/batch_size),
                  validation_data=gener(batch_size,DataName,EvalList,MIN,MAX),validation_steps=int(len(EvalList)/batch_size),callbacks=callbacks_list)
        model.save_weights(SaveName);
    else:
        model.load_weights(SaveName)
    return model 

def loadmodel(ModelType=3): # model type 3 seems to be the better one
    Path='Model'+str(ModelType)+'.h5';
    MIN,MAX=np.load('minmax.npy')    
    INPUT_SHAPE=[1,128,128,3];
    OUTPUT_SHAPE=[1,1515,1];
    model=modelmake(INPUT_SHAPE,OUTPUT_SHAPE,ModelType)
    model.load_weights(Path)
    return model 

def splitdata(List):
    N=np.int32([0,len(List)*.8,len(List)*.9,len(List)])
    TrainList=List[N[0]:N[1]]
    EvalList=List[N[1]:N[2]]
    TestList=List[N[2]:N[3]]
    return TrainList, EvalList, TestList
def testmodel(model,DataName,TestList,ModelType=3):
    MIN,MAX=np.load('minmax.npy') 
    G=gener(len(TestList),DataName,TestList,MIN,MAX)
    L=next(G)
    x=L[0]
    y=L[1]
    y2=model.predict(L[0])
    print('\n# Evaluate on '+ str(TestList.shape[0]) + ' test data')
    model.evaluate(x,y,batch_size=50)
    #  Denormalize the predictions
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y2=np.multiply(y2,(MAX-MIN))+MIN
    y[:,0:15]=10**y[:,0:15]
    y2[:,0:15]=10**y2[:,0:15]

    # save test results as mat file for postprocessing with matlab
    import scipy.io as sio
    sio.savemat('Tested_Data_Model'+str(ModelType)+'.mat',{'y':y,'y2':y2})
    # #  Show prediction of 15 single-value features
    fig=plt.figure(figsize=(30,40))
    plt.rcParams.update({'font.size': 30})
    with open('VarNames.txt') as f:
        VarNames = list(f)
    for I in range(15):
        ax = fig.add_subplot(5,3,I+1)
        X=y[:,I]
        Y=y2[:,I]
        plt.scatter(X,Y)
        plt.ylabel('Predicted')
        plt.xlabel('Ground truth')
        plt.tick_params(direction="in")
        plt.text(.5,.9,VarNames[I],horizontalalignment='center',transform=ax.transAxes)
        plt.xlim(np.min(X),np.max(X))
        plt.ylim(np.min(Y),np.max(Y))
        if I==0:
            ax.set_yscale('log')
            ax.set_xscale('log')
    plt.savefig('images/Single-value_Features.png')
def mat2np(Name): # load the MATLAB array as numpy array
    B=sio.loadmat(Name)  
    return B['A']    
def slicevol(A):
    A=np.squeeze(A)
    B=np.zeros((1,A.shape[0],A.shape[1],3))
    B[0,:,:,0]=A[int(A.shape[0]/2),:,:]
    B[0,:,:,1]=A[:,int(A.shape[1]/2),:]
    B[0,:,:,2]=A[:,:,int(A.shape[2]/2)]
    return B
def feedsampledata(A=None,FileName=None):
    if FileName!=None:
        extention=os.path.splitext(FileName)[1]
    else:
        extention=''    
    if extention=='.mat':
        # A=mat2np(FileName)
        import scipy.io as sio
        A=sio.loadmat(FileName)['A'] 
    if extention=='.npy':
        A=np.load(FileName)
    if extention=='.npz':
        A=np.load(FileName)['A']   
    if extention=='.npy' or extention=='.mat' or extention=='.npz' or FileName==None:    
        if len(A.shape)==3:
            
            A=np.int8(A!=0)   
            LO,HI=makeblocks(A.shape,w=256,ov=.1)
            N=len(HI[0])*len(HI[1])*len(HI[2]) # number of subsamples
            AA=np.zeros((N,256,256,3))
            a=0
            for I in range(len(LO[0])):
                for J in range(len(LO[1])):
                    for K in range(len(LO[2])):
                        temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J],LO[2][K]:HI[2][K]]
                        temp1=np.squeeze(temp[int(temp.shape[0]/2),:,:]);
                        temp2=np.squeeze(temp[:,int(temp.shape[1]/2),:]);
                        temp3=np.squeeze(temp[:,:,int(temp.shape[2]/2)]);
                        AA[a,...]=np.stack((temp1,temp2,temp3),axis=2)
                        a=a+1        
    if extention=='.png' or extention=='.jpg' or extention=='.bmp':
        A=plt.imread(FileName)
        if len(A.shape)!=2:
            print('Converting image to grayscale...')
            A=np.mean(A,axis=2)
            print('Converting image to binary...')
            import cv2
            ret,A = cv2.threshold(A,127,255,cv2.THRESH_BINARY)
        A=np.int8(A!=0)   
        LO,HI=makeblocks(A.shape,w=256,ov=.1)
        N=len(HI[0])*len(HI[1]) # number of subsamples
        AA=np.zeros((N,256,256,3))
        a=0
        for I in range(len(LO[0])):
            for J in range(len(LO[1])):
                temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J]]
                AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)
                a=a+1
    if FileName==None: 
        if len(A.shape)==2:
            A=np.int8(A!=0)   
            LO,HI=makeblocks(A.shape,w=256,ov=.1)
            N=len(HI[0])*len(HI[1]) # number of subsamples
            AA=np.zeros((N,256,256,3))
            a=0
            for I in range(len(LO[0])):
                for J in range(len(LO[1])):
                    temp=A[LO[0][I]:HI[0][I],LO[1][J]:HI[1][J]]
                    AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)
                    a=a+1
    
    B=ecl_distance(AA)
    return B

def writeh5slice(A,FileName,FieldName,Shape):
    # example: writeh5slice(A,'test3.h5','X',Shape=[70,70,1])
    D=len(Shape)  
    if D==2:
         maxshape=(None,Shape[0],Shape[1])
         Shape0=(1,Shape[0],Shape[1])
         A=np.reshape(A,Shape0)
    if D==3:
         maxshape=(None,Shape[0],Shape[1],Shape[2])
         Shape0=(1,Shape[0],Shape[1],Shape[2])
         A=np.reshape(A,Shape0)
    if D==4:
         maxshape=(None,Shape[0],Shape[1],Shape[2],Shape[3])  
         Shape0=(1,Shape[0],Shape[1],Shape[2],Shape[3])
         A=np.reshape(A,Shape0)
    try:
        with h5py.File(FileName, "r") as f:
            arr=f[FieldName]
        with h5py.File(FileName, "a") as f:
            arr=f[FieldName]
            Slice=arr.shape[0]
            arr.resize(arr.shape[0]+1, axis=0)
            arr[Slice,...]=A
        print('writing slice '+ str(Slice))
    except:
        with h5py.File(FileName, "a") as f:
            f.create_dataset(FieldName, Shape0,maxshape=maxshape, chunks=True,dtype=A.dtype,compression="gzip", compression_opts=5)
            f[FieldName][0,...]=A   
def normalize(A):
    A_min = np.min(A)
    return (A-A_min)/(np.max(A)-A_min)        
def predict(model,A,res=5):
    MIN,MAX=np.load('minmax.npy')    
    y=model.predict(A)
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y[:,0:15]=10**y[:,0:15]
    # y[:,1]=10**y[:,1]
    y=np.mean(y,axis=0)
    val=y[0:15]
    val[0]=val[0]*res*res
    val[3]=val[3]/res/res/res
    val[10]=val[10]/res
    val[6]=val[6]*res
    val[7]=val[7]*res
    val[8]=val[8]*res
    val[13]=val[13]*res
    d=100
    output=val
    for I in range(15):
        func=y[I*d+15:(I+1)*d+15]
        if I in [19,20,21,22,23,24,29]:
            func=func*res
        if I in [18]:
            func=func/res
        if I in [25]:
            func=func*res*res                
        output=np.append(output,func)
    return output
def ecl_distance(A):
    B=np.zeros((A.shape[0],128,128,3))
    for I in range(A.shape[0]):
        for J in range(A.shape[3]):
            t=distance(np.squeeze(1-A[I,:,:,J]))-distance(np.squeeze(A[I,:,:,J]))
            # t=normalize(t)
            t=np.float32(t)/64
            
            t[t>1]=1
            t[t<-1]=-1
            
            t = MaxPooling2D((2, 2)) (np.reshape(t,(1,256,256,1)))
            t=np.float64(t)
            B[I,:,:,J]=np.squeeze(t)
    return B
def normal(A):
    A_min = np.min(A)
    return (A-A_min)/(np.max(A)-A_min) 
def show_feature_maps(A):
    N=np.ceil(np.sqrt(A.shape[0]))
    f=plt.figure(figsize=(N*10,N*10))
    for I in range(A.shape[0]):
        plt.subplot(N,N,I+1)
        plt.imshow(normal(np.squeeze(A[I,:,:,:])))  
        plt.axis('off')
    plt.show()
    
    f.savefig('images/initial_feature_maps.png')
def makeblocks(SS,n=None,w=None,ov=0):
    # w is the fixed width of the blocks and n is the number of blocks
    # if the number be high while w is fixed, blocks start to overlap and ov is between 0 to 1 gets desired overlapping degree
    # example:dp.makeblocks([100,200],w=16,ov=.1)
    HI=[]
    LO=[]
    for S in SS:
        if w==None and n!=None:
            mid=np.ceil(np.linspace(0,S,n+1));
            lo=mid; lo=np.delete(lo,-1);
            hi=mid; hi=np.delete(hi,0);
        if w!=None and n!=None:
            mid=np.ceil(np.linspace(0,S-w,n));
            lo=mid; 
            hi=mid+w
        if w!=None and n==None: # good for image translation
            mid=np.asarray(np.arange(0,S,int(w*(1-ov))))
            lo=mid; 
            hi=mid+w    
            p=np.argwhere(hi>S)
            if len(p)>0:
                diff=hi[p]-S
                hi[p]=S
                lo[p]=lo[p]-diff
                hi=np.unique(hi)
                lo=np.unique(lo)
        HI.append(hi)
        LO.append(lo)        
    return LO,HI       
def prettyresult(vals,FileName,units='um',verbose=1):
    vals=np.squeeze(vals)
    with open('VarNames.txt') as f:
        VarNames = list(f)
    b=np.round(vals[0:15],7)
    f = open(FileName, 'w')
    f.write('DeePore output results including 15 single-value' +'\n'+ 'paramters, 4 functions and 11 distributions'+'\n')
    f.write('_' * 50+'\n')
    f.write('        ### Single-value parameters ###'+'\n')
    f.write('_' * 50+'\n')
    f.write('\n')
    t='Properties'
    spa=' ' * (40-len(t))
    f.write(t+spa+'Value'+'\n')
    f.write('-' * 50+'\n')
    for i in range(len(b)):
        t=VarNames[i].strip()
        if units=='um':
            t=t.replace('px','um')
        spa=' ' * (40-len(t))
        results=t +spa+str(b[i])+'\n'
        f.write(results)
    f.write('\n')
    f.write('_' * 50+'\n')
    f.write('       ### Functions and distributions ###'+'\n')
    f.write('_' * 50+'\n')
    for I in range(15):
        multiplier=1
        t=VarNames[I+15].strip()
        if units=='um':
            t=t.replace('px','um')
        f.write('\n')
        f.write('\n')
        f.write('# '+t+'\n')    
        f.write('-' * 50+'\n')
        xlabel='Cumulative probability'
        if I+15 in [15,16,17]:
            xlabel='Wetting-sat (Sw)'
        if I+15 in [18]:
            xlabel='lag (px)'
            multiplier=50
        spa=' ' * (40-len(xlabel))
        f.write(xlabel+spa+'Value'+'\n')
        f.write('-' * 50+'\n')
        shift=I*100+15
        for J in range(100):
            t=str(np.round((J*.01+.01)*multiplier,2))
            spa=' ' * (40-len(t))
            f.write(t+spa+str(np.round(vals[J+shift],7))+'\n')
    f.close()
    a=0
    if verbose:
        print('\n')
        with open(FileName,"r") as f:
            for line in f:
                print(line)
                a=a+1
                if a>23:
                    print('-' * 50+'\n')
                    print('To see all the results please refer to this file: \n')
                    print(FileName+'\n')
                    break
def readh5slice(FileName,FieldName,Slices):
    with h5py.File(FileName, "r") as f:
         A=f[FieldName][np.sort(Slices),...]
    return A
def create_compact_dataset(Path_complete,Path_compact):
    S=hdf_shapes(Path_complete,['X'])
    for I in range(S[0][0]):
        X=readh5slice(Path_complete,'X',[I])
        Y=readh5slice(Path_complete,'Y',[I])
        X=slicevol(X)
        X=ecl_distance(X)
        writeh5slice(X,Path_compact,'X',Shape=[128,128,3])
        writeh5slice(Y,Path_compact,'Y',Shape=[1515,1])


def showentry(A):
    """shows 3 slices of a volume data """
    A=np.squeeze(A)
    plt.figure(num=None, figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')
    # CM=plt.cm.jet
    # CM=plt.cm.plasma
    CM=plt.cm.viridis
    ax1=plt.subplot(1,3,1); plt.axis('off'); ax1.set_title('X mid-slice')
    plt.imshow(np.squeeze(A[np.int(A.shape[0]/2), :,:]), cmap=CM, interpolation='nearest')
    # plt.colorbar(orientation="horizontal")
    ax2=plt.subplot(1,3,2); plt.axis('off'); ax2.set_title('Y mid-slice')
    plt.imshow(np.squeeze(A[:,np.int(A.shape[1]/2), :]), cmap=CM, interpolation='nearest')
    # plt.colorbar(orientation="horizontal")
    ax3=plt.subplot(1,3,3); plt.axis('off'); ax3.set_title('Z mid-slice'); 
    plt.imshow(np.squeeze(A[:,:,np.int(A.shape[2]/2)]), cmap=CM, interpolation='nearest')
    # plt.colorbar(orientation="horizontal")        
    plt.savefig('images/First_entry.png')
def parfor(func,values):
    # example 
    # def calc(I):
    #     return I*2
    # px.parfor(calc,[1,2,3])
    N=len(values)
    from joblib import Parallel, delayed
    from tqdm import tqdm
    Out = Parallel(n_jobs=-1)(delayed(func)(k) for k in tqdm(range(1,N+1)))
    return Out