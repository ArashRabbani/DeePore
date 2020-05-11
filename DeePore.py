import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
import os, sys
import matplotlib.pyplot as plt

def check_get(url,File_Name): 
    from urllib.request import urlretrieve   
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
        if ans=='Y':
            print('Beginning file download. This might take several minutes.')
            urlretrieve(url,File_Name,download_callback)   
    else:
        print('File "' +File_Name +'" is detected on your machine.'  )
def makemap(A):
    # A is a binary 3D array with zero as void space and 1 as solid space
    import skimage
    import scipy.ndimage as ndi
    A1=np.squeeze(A[:,:,int(A.shape[2]/2)])
    A2=np.squeeze(A[:,int(A.shape[1]/2),:])
    A3=np.squeeze(A[int(A.shape[0]/2),:,:])
    A1 = ndi.distance_transform_edt(1-A1)
    A2 = ndi.distance_transform_edt(1-A2)
    A3 = ndi.distance_transform_edt(1-A3)
    A1 = A1/np.max(A1)
    A2 = A2/np.max(A2)
    A3 = A3/np.max(A3)
    A1=skimage.measure.block_reduce(A1, (2,2), np.max)
    A2=skimage.measure.block_reduce(A2, (2,2), np.max)
    A3=skimage.measure.block_reduce(A3, (2,2), np.max)
    B=np.uint8(np.stack((A1,A2,A3),axis=2)*255)
    plt.imshow(B)
def DeePore1(INPUT_SHAPE,OUTPUT_SHAPE):
    # Charactrization of the macroscopic properties of porous material
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
    model.summary()
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
            D=int(np.sum(np.isnan(y)))+int(np.sum(np.isinf(y)))+int(y[1]>120)+int(y[4]>1.7)+int(y[0]<1e-4)+int(y[2]<1e-5)+int(y[14]>.7)
#            D=int(y[1]>120)+int(y[4]>1.7)
#            D=int(y[0]<1e-4)+int(y[2]<1e-5)
#            D=int(y[14]>.7)
            # D=D+int(np.sum(np.isnan(y)))
            if D>0:
                pass
            else:
                List=np.append(List,counter)
                y[0]=np.log10(y[0]) # applying log10 on absolute permeability values
                maxid=np.argwhere(y>MAX)
                minid=np.argwhere(y<MIN)
                MAX[maxid[:,0]]=y[maxid[:,0]]
                MIN[minid[:,0]]=y[minid[:,0]]
            counter=counter+1
        Singles=15
        for I in range(15):
            MAX[Singles+100*I:Singles+100*(I+1)]=np.max(MAX[Singles+100*I:Singles+100*(I+1)])
            MIN[Singles+100*I:Singles+100*(I+1)]=np.min(MIN[Singles+100*I:Singles+100*(I+1)])
    return List,MIN,MAX
def gener(batch_size,Data,List,MIN,MAX):
    with h5py.File(Data,'r') as f:
        length=len(List)
        samples_per_epoch = length
        number_of_batches = int(samples_per_epoch/batch_size)
        counter=0
        Sample_Weights=[None]
        while 1:
            t1=f['X'][np.sort(List[batch_size*counter:batch_size*(counter+1)]),...]
            t2=f['Y'][np.sort(List[batch_size*counter:batch_size*(counter+1)]),...]

            # t1=f['X'][List[batch_size*counter:batch_size*(counter+1)],...]
            # t2=f['Y'][List[batch_size*counter:batch_size*(counter+1)],...]
            X_batch=t1.astype('float32')/255
            y_batch=t2.astype('float32')
            y_batch=np.reshape(y_batch,(y_batch.shape[0],y_batch.shape[1]))
            y_batch[:,0]=np.log10(y_batch[:,0])
            Min=np.tile(np.transpose(MIN),(batch_size,1))
            Max=np.tile(np.transpose(MAX),(batch_size,1))
            y_batch=(y_batch-Min)/(Max-Min)
            counter += 1
            yield X_batch,y_batch, Sample_Weights  
            if counter >= number_of_batches: #restart counter to yeild data in the next epoch as well
                counter = 0
def hdf_shapes(Name,Fields):
    # Fields is tuple of hdf file fields
    Shape = [[] for _ in range(len(Fields))]
    with h5py.File(Name, 'r') as f:
        for I in range(len(Fields)):
            Shape[I]=f[Fields[I]].shape  
    return Shape                
def trainmodel(DataName,TrainList,EvalList,MIN,MAX,retrain=0):
    if retrain !=0:
        try:
            tf.compat.v1.disable_v2_behavior()
        except:
            pass
    SaveName='Model.h5';
    INPUT_SHAPE,OUTPUT_SHAPE =hdf_shapes(DataName,('X','Y')); 
    model=DeePore1(INPUT_SHAPE,OUTPUT_SHAPE)
    batch_size=10     
    if retrain:
        model.fit(gener(batch_size,DataName,TrainList,MIN,MAX), epochs=100,steps_per_epoch=int(len(TrainList)/batch_size),
                  validation_data=gener(batch_size*2,DataName,EvalList,MIN,MAX),validation_steps=int(len(EvalList)/batch_size/2))
        model.save_weights(SaveName);
    else:
        model.load_weights(SaveName)
    
    return model 
def splitdata(List):
    N=np.int32([0,len(List)*.64,len(List)*.8,len(List)])
    TrainList=List[N[0]:N[1]]
    EvalList=List[N[1]:N[2]]
    TestList=List[N[2]:N[3]]
    return TrainList, EvalList, TestList
def testmodel(model,DataName,TestList,MIN,MAX):
    G=gener(len(TestList),DataName,TestList,MIN,MAX)
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
    # #  Show prediction of 15 single-value features
    fig=plt.figure(figsize=(30,40))
    plt.rcParams.update({'font.size': 30})
    with open('VarNames.txt') as f:
        VarNames = list(f)
    # df = pd.read_excel('VarNames.xlsx')
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
    import scipy.io as sio
    B=sio.loadmat(Name)  
    return B['A']    
def slicevol(A):
    A=np.squeeze(A)
    B=np.zeros((1,A.shape[0],A.shape[1],3))
    B[0,:,:,0]=A[int(A.shape[0]/2),:,:]
    B[0,:,:,1]=A[:,int(A.shape[1]/2),:]
    B[0,:,:,2]=A[:,:,int(A.shape[2]/2)]
    return B
def readsampledata(FileName='Data/Sample.mat'):
    import os
    import cv2
    extention=os.path.splitext(FileName)[1]
    if extention=='.mat':
        A=mat2np(FileName)
    if extention=='.npy':
        A=np.load(FileName)
    if extention=='.npz':
        A=np.load(FileName)['A']   
    if extention=='.npy' or extention=='.mat' or extention=='.npz':    
        A=np.int8(A!=0)   
        LO,HI=makeblocks(A.shape,w=256,ov=.1)
        N=len(HI[0])*len(HI[1])*len(HI[2]) # number of subsamples
        AA=np.zeros((N,256,256,3))
        a=0
        for I in range(len(LO[0])):
            for J in range(len(LO[1])):
                for K in range(len(LO[2])):
                    temp=A[LO[0][I]:HI[0][I],LO[1][I]:HI[1][I],LO[2][K]:HI[2][K]]
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
            ret,A = cv2.threshold(A,127,255,cv2.THRESH_BINARY)
        A=np.int8(A!=0)   
        LO,HI=makeblocks(A.shape,w=256,ov=.1)
        N=len(HI[0])*len(HI[1]) # number of subsamples
        AA=np.zeros((N,256,256,3))
        a=0
        for I in range(len(LO[0])):
            for J in range(len(LO[1])):
                temp=A[LO[0][I]:HI[0][I],LO[1][I]:HI[1][I]]
                AA[a,...]=np.stack((temp,np.flip(temp,axis=0),np.flip(temp,axis=1)),axis=2)
                a=a+1
    return AA
def writeh5slice(A,FileName,FieldName,Shape):
    # example: writeh5slice(A,'test3.h5','X',Shape=[70,70,1])
    import h5py
    import numpy as np
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
def predict(model,A,MIN,MAX,res=5):
    y=model.predict(A)
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y[:,0]=10**y[:,0]
    y=np.mean(y,axis=0)
    with open('VarNames.txt') as f:
        VarNames = list(f)
        val=y[0:15]
        val[0]=val[0]*res*res
        val[3]=val[3]/res/res/res
        val[10]=val[10]/res
        val[6]=val[6]*res
        val[7]=val[7]*res
        val[8]=val[8]*res
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
        # val=[VarNames,val]
    return output
    # plt.imsave('Data/Sample.png',np.squeeze(A[:,:,128]),cmap='gray')
    # plt.imsave('Data/Sample.jpg',np.squeeze(A[:,:,128]),cmap='gray')
def maxpool2(A):
    from scipy.ndimage.filters import maximum_filter
    # C=np.zeros((A.shape[0]+1,A.shape[1]+1))
    # C[1:,1:]=A
    # A=C
    print(A)
    B=maximum_filter(A,footprint=np.ones((2,2)))
    print(B)
    B=np.delete(B, range(1, B.shape[0], 2), axis=0)  
    B=np.delete(B, range(1, B.shape[1], 2), axis=1)
    return B
def ecl_distance(A):
    tf.compat.v1.enable_v2_behavior()
    B=np.zeros((A.shape[0],128,128,3))
    from scipy.ndimage import distance_transform_edt as distance
    for I in range(A.shape[0]):
        for J in range(A.shape[3]):
            t=distance(np.squeeze(A[I,:,:,J]))
            t=normalize(t)
            t=np.float32(t)
            t = MaxPooling2D((2, 2)) (np.reshape(t,(1,256,256,1)))
            t=np.float64(t)
            B[I,:,:,J]=np.squeeze(t)
    return B
def show_feature_maps(A):
    N=np.ceil(np.sqrt(A.shape[0]))
    plt.figure(figsize=(N*10,N*10))
    for I in range(A.shape[0]):
        plt.subplot(N,N,I+1)
        plt.imshow(np.squeeze(A[I,:,:,:]))  
        plt.axis('off')
    plt.show()
def makeblocks(SS,n=None,w=None,ov=0):
    # w is the fixed width of the blocks and n is the number of blocks
    # if the number be high while w is fixed, blocks start to overlap and ov is between 0 to 1 gets desired overlapping degree
    # example:dp.makeblocks([100,200],w=16,ov=.1)
    import numpy as np
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
def prettyresult(vals,FileName,units='um'):
    # import numpy as np
    # vals=np.random.rand(1515)
    with open('VarNames.txt') as f:
        VarNames = list(f)
    # a=VarNames
    b=np.round(vals[0:15],7)
    f = open(FileName, 'w')
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
        
    
    for I in range(15):
        f.write('\n')
        f.write('-' * 50+'\n')
        f.write('# '+VarNames[I+15])    
        f.write('-' * 50+'\n')
        t='Wetting-sat (Sw)'
        spa=' ' * (40-len(t))
        f.write(t+spa+'Value'+'\n')
        for J in range(10):

            t=str(J*.1)
            spa=' ' * (40-len(t))
            f.write(t+spa+str(vals[J])+'\n')
    
    f.close()
def readh5slice(FileName,FieldName,Slices):
    # example: B=px.readh5slice('test3.h5','X',[1,2])
    import numpy as np
    import h5py
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
        X=np.uint8(X*255)
        writeh5slice(X,Path_compact,'X',Shape=[128,128,3])
        writeh5slice(Y,Path_compact,'Y',Shape=[1515,1])
        
    
    
