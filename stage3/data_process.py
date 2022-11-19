from sklearn.model_selection import train_test_split
import pandas as pd
#将数据集划分为训练集、测试集
#由于本人水平太菜了故只用了前三个数据没有使用tag数据

def spilt_data():
    x= original_data.iloc[:,:] # 选取 data 所有行、所有列数据
    y = original_data.iloc[:,0] # 选取 data 所有行、第一列数据
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    X_test=X_test.drop(['time','tag'],axis=1)
    X_train=X_train.drop(['time','tag'],axis=1)
    X_train.to_csv("train.csv",index=False,sep=',')
    X_test.to_csv("test.csv",index=False,sep=',')
    # print(original_data.size,train_data.size,test_data.size)
#对数据集和测试机按照用户id进行排序
def sort(df_loc,name):
    df_loc = df_loc.sort_values(axis=0, by=['user_id','movie_id'], ascending=True)
    # print('df_loc=', df_loc)
    df_loc.to_csv(name,header=True,index=False)
    # print(pd.read_csv('train.csv').size)

def csv_to_txt(file):
    new_file=file[:-4]+'.txt'
    with open(new_file,'w', encoding='utf-8') as f:
        for line in pd.read_csv(file).values:
            #str(line[0])：csv中第0列；+','+：csv两列之间保存到txt用逗号（，）隔开；'\n'：读取csv每行后在txt中换行
            f.write((str(line[0])+','+str(line[1])+','+str(line[2])+'\n'))
        content=open(new_file).read()
        f.seek(0, 0)
        f.write('user_id,movie_id,movie_score\n'+content)

if __name__=='__main__':
    original_data=pd.read_csv('Movie_score.csv')
    spilt_data()
    train_data=pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')
    sort(train_data,'train.csv')
    sort(test_data,'test.csv')
    # csv_to_txt('train.csv')
    # csv_to_txt('test.csv')