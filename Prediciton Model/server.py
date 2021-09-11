
import time
import sys
import joblib
from keras.models import load_model
import keras.backend as K
from xmlrpc.server import SimpleXMLRPCServer
import traceback
import numpy as np

def R_squared(y, y_pred):
    res =  K.sum(K.square(y - y_pred)) 
    tot = K.sum(K.square(y - K.mean(y))) 
    return ( 1 - res/(tot + K.epsilon()) )
def prmse(y, y_pred):
    rmse = K.mean(K.square(y-y_pred))**0.5
    Tmav = K.mean(y)
    return rmse/Tmav*100

try:
    print(r'loading model')
    model_path = ['D:/Desktop/acadia/Prediciton Model/keras_load_1win_pos_R0.99847_RMSE4.43.h5', 'D:/Desktop/acadia/Prediciton Model/keras_load_2win_pos_R0.99751_RMSE5.56.h5', 'D:/Desktop/acadia/Prediciton Model/keras_load_3win_pos_R0.99682_RMSE6.30.h5']
    models=[load_model(p,custom_objects={"R_squared":R_squared,"prmse":prmse}) for p in model_path]
except Exception as e:
    print(e)
    print(r'loading model FAIL')


scalers=[joblib.load(p+'.scalar') for p in model_path]
def predict_resolution(X):
    X=X.reshape([-1,X.shape[-1]])
    idx=int(X[:,-1].mean())-1
    X = X[:,:-1]
    rx=int(X[:,-2].mean())
    ry=int(X[:,-1].mean())
    sx=1/rx
    sy=1/ry
    row_num=X.shape[0]
    inputs=np.zeros([rx*ry,2])
    for ix in range(rx):
        for iy in range(ry):
            inputs[ix*ry+iy] = np.array([sx*ix+sx/2,sy*iy+sy/2])
    inputs=np.tile(inputs,[row_num,1])
    fullinput = np.hstack([np.repeat(X[:,:-2],rx*ry,0),inputs])
    return models[idx].predict(scalers[idx].transform(fullinput)).transpose().flatten()

def predict():
    try:
        #v=np.loadtxt(r"D:/Desktop/acadia/Prediciton Model/temp.txt",delimiter=',',dtype=float)
        #if len(v.shape)==1: v=np.expand_dims(v, axis=0)
        #res=np.apply_along_axis(predict_resolution,1,v)
        res=[]
        with open(r"D:/Desktop/acadia/Prediciton Model/res.txt",'w+') as outf:
            with open(r"D:/Desktop/acadia/Prediciton Model/temp.txt",'r') as f:
                for l in f:
                    inp = np.array([float(d) for d in l.split(",")])
                    outp= predict_resolution(inp).tolist()
                    res.append(outp)
                    outf.write(",".join([str(p) for p in outp]))
                    outf.write("\n")
        #np.savetxt(r"D:/Desktop/acadia/Prediciton Model/res.txt",res,fmt="%.1f")
        return r"D:/Desktop/acadia/Prediciton Model/res.txt"
    except Exception as e:
        print(e)
        traceback.print_exc()
    #print('predict:', res)

def execute(cmd):
    exec(cmd)
def check():
    print('server start!')
    return True
server = SimpleXMLRPCServer(("127.0.0.1", 666))
print ("Listening on port 666...")
server.register_function(predict, "predict")
server.register_function(execute, "execute")
server.register_function(check, "check")
server.serve_forever()
print('server start!')
