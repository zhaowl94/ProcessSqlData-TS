# -*- coding: utf8 -*-
import datetime
import gatspy.periodic
import math
import matplotlib
import matplotlib.dates
import matplotlib.pyplot
import numpy
import os
import pandas
import pp
import psycopg2
import pylab
import re
import scipy.interpolate
import shutil
import sklearn.linear_model
import statsmodels.stats.diagnostic
import statsmodels.tsa.arima_model
import statsmodels.tsa.stattools 
import sys
import time

# ------#
# part1 #
# ------#
# 从目标数据库（psycopg2）下载数据，数据存到csv表格中
if os.path.exists(u'../DataRaw'):
    shutil.rmtree(u'../DataRaw')
    os.mkdir(u'../DataRaw')
else:
    os.mkdir(u'../DataRaw')

# 连接数据库
conn = psycopg2.connect(database="jarolweb", user="user_analysis1", \
                    password="dividend-opacity-resolved-umbilici-lame-antacid", host="60.190.152.198", \
                    port="35432")

# 新建一个光标
cur = conn.cursor()

CurOffset = 0
CurLength = 10000
DataFileCount = 0

cur.execute("select column_name from information_schema.columns where table_name = 'dashboard_dtudata' ")

# SqlTitle是一个列矩阵
SqlTitle = cur.fetchall()
SqlTitle = numpy.array(SqlTitle)
SqlTitle = SqlTitle.transpose()

while True:
# def ProcessFile()
    DataFileCount = DataFileCount + 1
    CurOffset = CurOffset + CurLength
    cur.execute("select * from dashboard_dtudata limit %d offset %d;" % (CurLength,CurOffset))
    # cur.execute("select * from dashboard_dtudata limit 5 offset 10;")

    SqlNum = cur.fetchall()        # all rows in cur

    SqlNum = numpy.array(SqlNum)

    df = pandas.DataFrame(SqlNum,columns = SqlTitle.tolist())
    outfname = '../DataRaw/SqlData' + str(DataFileCount) + '.csv'
    uifname = unicode(outfname,'utf8')
    df.to_csv(uifname)
    # workbook.save(uifname)

    if SqlNum.shape[0] < CurLength:
        break

# ----- #
# part2 #
# ----- #
# 把按条目顺序（？）存储的csv数据，整理成按照元件存储的csv数据，不解析元件特征。
# 总数据量大，所以分多次整理。整理结果暂存在csv文件中，减小内存消耗。
# 一次读写多个元件对应的csv，降低IO频率，提高速度。
inpath = '../DataRaw'
uipath = unicode(inpath, 'utf8')

pathfilelist = os.listdir(uipath)
# python的排序函数，可以自定义排序规则。x:后面是用于排序的规则。
pathfilelist.sort(key= lambda x:int(x[7:-4]))
# 文件排序后，读入的数据是按时间顺序的。便于下一步的处理。

# SqlId是储存处理过程中，已经出现过的，元件的编号
SqlId = numpy.array([[]])

if os.path.exists(u'../DataProcess1'):
    # 只能删除空目录
    # os.rmdir(u'../DataProcess1')
    shutil.rmtree(u'../DataProcess1')
    os.mkdir(u'../DataProcess1')
else:
    os.mkdir(u'../DataProcess1')

time1 = time.time()
count_iter = 0
for i_iter in range (0,pathfilelist.__len__(),100):
# for i_filename in pathfilelist:
    df1 = pandas.DataFrame()
    for j_iter in range (0,100):
        if count_iter == pathfilelist.__len__():
            break
        uifname = unicode(inpath, 'utf8') + unicode('/', 'utf8') + pathfilelist[count_iter]
        # print uifname
        df1 = pandas.concat([df1,pandas.read_csv(uifname)])
        count_iter = count_iter + 1

    # 分批写入csv
    # 识别出当前csv表格数据涉及的元件编号
    TempSqlId = numpy.array(numpy.unique(df1['sn']))
    # 分元件号写入csv
    for i in range(0, len(TempSqlId)):
        # 如果已经处理过的csv的表格数据涉及的元件编号不包含当前csv表格数据涉及的元件编号
        if not TempSqlId[i] in SqlId:
            # 记录已经处理过的csv的表格数据涉及的元件编号
            SqlId = numpy.column_stack((SqlId, TempSqlId[i]))
            # 新建一个csv表格，存储处理过的数据
            savename = u'../DataProcess1/' + str(TempSqlId[i]).encode('utf-8') + u'.csv'
            Tempdf1 = df1[df1['sn'] == TempSqlId[i]]
            Tempdf1.to_csv(savename)
        else:
            uifname1 = u'../DataProcess1/' + str(TempSqlId[i]).encode('utf-8') + u'.csv'
            df2 = pandas.read_csv(uifname1)
            df2 = df2.iloc[:,1:6]
            try:
                Tempdf2 = pandas.concat([df2,df1[df1['sn'] == TempSqlId[i]].iloc[:,1:6]],ignore_index=True)
            except:
                print df2.shape
                print df1[df1['sn'] == TempSqlId[i]].iloc[:,1:6].shape
            savename = u'../DataProcess1/' + str(TempSqlId[i]).encode('utf-8') + u'.csv'
            Tempdf2.to_csv(savename)

    time2 = time.time()
    print time2 - time1

# ----- #
# part3 #
# ----- #
# 处理按元件存储的csv文件，解析元件特征。
# 每个元件的解析结果整理成一张表，行索引是时刻，列索引是元件特征。
# 多进程，每次集中处理一个文件，每个文件有10000行数据。
if os.path.exists(u'../DataProcess2'):
    shutil.rmtree(u'../DataProcess2')
    os.mkdir(u'../DataProcess2')
else:
    os.mkdir(u'../DataProcess2')

inpath = '../DataProcess1'
uipath = unicode(inpath, 'utf8')

pathfilelist = os.listdir(uipath)

for i_filename in pathfilelist:
    infname = unicode('../DataProcess1', 'utf-8') + u'/' + i_filename
    df0 = pandas.read_csv(infname)
    LengthOfFile = df0.shape[0]

    if os.path.exists(u'../DataProcess2' + u'/' + i_filename[:-4]):
        shutil.rmtree(u'../DataProcess2' + u'/' + i_filename[:-4])
        os.mkdir(u'../DataProcess2' + u'/' + i_filename[:-4])
    else:
        os.mkdir(u'../DataProcess2' + u'/' + i_filename[:-4])

    FileName = i_filename

    # def ProcessFile(FileName , df0 , iPart):
    for iPart in range(0 , int(math.ceil(LengthOfFile/10000))):
        # df0是从csv文件读入的完整dataframe
        LengthOfFile = df0.shape[0]
        if 10000 * iPart >= LengthOfFile:
            df1 = df0.iloc[10000 * iPart : , 1:]
        else:
            df1 = df0.iloc[10000 * iPart : 10000 * iPart + 10000, 1:]
        
        df3 = pandas.DataFrame()

        m1 = re.compile('u\'[\w\W]*\':u\'[\w\W]*\'')
        m2 = re.compile('\"[\w\W]*\":\"[\w\W]*\"')
        for i_row in range(0, df1.__len__()):
            TempRowData = df1.iloc[i_row, :]
            # a1 = TempRowData.iloc[2][1:-1].replace(' ', '')
            # 用column名索引，最靠谱了！！！
            a1 = TempRowData['data'][1:-1].replace(' ', '')
            a2 = a1.split(',')
            a5 = []
            if m1.match(a2[0]):
                for a3 in a2:
                    a4 = [a3.split(':')[0][2:-1], a3.split(':')[1][2:-1]]
                    a5.append(a4)
            elif m2.match(a2[0]):
                for a3 in a2:
                    a4 = [a3.split(':')[0][1:-1], a3.split(':')[1][1:-1]]
                    a5.append(a4)
            if len(a5) == 0:
                continue
            a6 = numpy.array(a5)
            a7 = numpy.transpose(a6[:, 0])
            a8 = numpy.column_stack([numpy.array([numpy.array(TempRowData['sn'])]), \
                                     numpy.array([numpy.array(TempRowData['date'])]), \
                                     numpy.array([numpy.transpose(a6[:, 1])]), \
                                     numpy.array([numpy.array(TempRowData['slave_id'])]) ,\
                                     numpy.array([numpy.array(TempRowData['type_id'])])])

            df2 = pandas.DataFrame(a8, columns=['sn', 'date'] + a7.tolist() + ['slave_id', 'type_id'])
            # numpy.column_stack([numpy.array([numpy.array(TempRowData[0:2])]), a7, numpy.array([numpy.array(TempRowData[-2:])])])
            df3 = df3.append(df2)

        # df3 = df3.sort_values(by= 'date')
        outfname = unicode('../DataProcess2', 'utf-8') + u'/' + FileName[:-4] + u'/' + str(iPart) + u'.csv'
        df3.to_csv(outfname)

# ----- #
# part4 #
# ----- #
# 拼接分段处理的元件特征数据，数据按时间顺序排序。
if os.path.exists(u'../DataProcess3'):
    shutil.rmtree(u'../DataProcess3')
    os.mkdir(u'../DataProcess3')
else:
    os.mkdir(u'../DataProcess3')

uipath = u'../DataProcess2'
# uipath = unicode(inpath, 'utf8')

pathdirlist = os.listdir(uipath)

# for ListName in pathdirlist:
def ProcessFile(ListName):
    inpath = unicode('../DataProcess2/','utf-8') + ListName
    # inpath = u'../DataProcess2/' + ListName
    pathfilelist = os.listdir(inpath)
    if len(pathfilelist) == 0:
        return
        # continue
    df1 = pandas.DataFrame()
    for i_pathfile in pathfilelist:
        infname = inpath + u'/' + i_pathfile
        df2 = pandas.read_csv(infname)
        # df1.append(df2)必须用一个变量接才行！！！
        df1 = df1.append(df2)
    # pandas和numpy的处理结果都要用变量去接。和list.append不同，切记！！！！
    df1 = df1.sort_values(by= 'date')
    # unicode(,'utf-8')比u'创建Unicode字符串稳妥！！！
    # python2字符串太蛋疼了！！！
    # outfname = u'../DataProcess3/' + ListName + u'.csv'
    outfname = unicode('../DataProcess3/','utf-8') + ListName + unicode('.csv','utf-8')
    df1.to_csv(outfname)

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits 8 jobs and then retrieves the results
# inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
jobs = [job_server.submit(ProcessFile,(i_listname,), (), ("math","numpy","pandas","re")) for i_listname in pathdirlist]
for job in jobs:
    # print ("Task", i_filename, "is started")
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ----- #
# part5 #
# ----- #
# 把每个元件的每个特征存成一个csv文件。
# 每个csv文件存储一个时间序列，包含时刻、数据。
# 数据排除了空值，所以会是非等间隔采样的时间序列。
if os.path.exists(u'../DataProcess4'):
    shutil.rmtree(u'../DataProcess4')
    os.mkdir(u'../DataProcess4')
else:
    os.mkdir(u'../DataProcess4')


inpath = '../DataProcess3'
uipath = unicode(inpath, 'utf8')
pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
    infname = unicode('../DataProcess3', 'utf-8') \
    + u'/' + FileName
    # 读取原始数据的dataframe
    df1 = pandas.read_csv(infname)
    df1 = df1.iloc[:, 1:]
    # 特征数据的列的范围1:-4;date:-4;slave_id:-3;sn:-2;type_id:-1
    for df1col in df1.columns:
        # 每一个列数据的临时dataframe
        # 好像只能用.loc的方式选多轴数据
        if df1col not in ['sn','date','slave_id','type_id']:
            df2 = df1.loc[:, ['date', df1col]]
            # 提取出非nan的部分
            df2DataSeq = numpy.where(pandas.isnull(df2[df1col]) == False)
            # df2DataSeq是个tuple，df2DataSeq[0]是个array
            df3 = df2.iloc[df2DataSeq[0], :]
            df3DataSeq = []
            for i in range(0, df3.shape[0]):
                df3DataSeq.append(isinstance(df3.iloc[i, 1], float))
            # numpy.where输入的数据可以使series和ndarray，array貌似不行
            df3DataSeq = numpy.where(numpy.array(df3DataSeq) == True)
            df4 = df3.iloc[df3DataSeq[0], :]
            outfname = unicode('../DataProcess4', 'utf-8') + u'/' + FileName[:-4] + '_' + str(df1col) + '.csv'
            df4.to_csv(outfname)

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), ("math","numpy","pandas","re")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ----- #
# part7 #
# ----- #
# 剔除异常数据
if os.path.exists(u'../DataProcess5'):
    shutil.rmtree(u'../DataProcess5')
    os.mkdir(u'../DataProcess5')
else:
    os.mkdir(u'../DataProcess5')

# 源数据在ProcessedData3_2文件夹中
inpath = '../DataProcess4'
uipath = unicode(inpath,'utf-8')

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    infname = unicode('../DataProcess4', 'utf-8') \
    + u'/' + FileName
    df1 = pandas.read_csv(infname)
    if df1.shape[0] == 0:
        return
        # continue
    
    yDataSorted = df1.iloc[:,2].sort_values()
    yDataLength = df1.shape[0]
    yData25 = (yDataSorted[max(0 , int(numpy.floor(1./4 * yDataLength)))] + \
        yDataSorted[min(yDataLength-1 , int(numpy.ceil(1./4 * yDataLength)))])/2
    yData75 = (yDataSorted[max(0 , int(numpy.floor(3./4 * yDataLength)))] + \
        yDataSorted[min(yDataLength-1 , int(numpy.ceil(3./4 * yDataLength)))])/2
    yData50 = (yDataSorted[max(0 , int(numpy.floor(2./4 * yDataLength)))] + \
        yDataSorted[min(yDataLength-1 , int(numpy.ceil(2./4 * yDataLength)))])/2
    yDataUpLimit = yData75 + (yData75 - yData25) * 1.5
    yDataDownLimit = yData25 - (yData75 - yData25) * 1.5
    yDataUpTh = 62100
    # tData = df1.iloc[:,1].tolist()
    yData = df1.iloc[:,2]
    
    yDataRemainSeq = list(set(list(set(numpy.array(numpy.where(yData <= yDataUpLimit)). \
        tolist()[0]).intersection(set(numpy.array(numpy.where(yData >= yDataDownLimit)).tolist()[0])))). \
        intersection(set(numpy.array(numpy.where(yData <= yDataUpTh)).tolist()[0])))
    # yDataRemainSeq = numpy.array(numpy.where(yData <= yDataUpLimit and yData >= yDataDownLimit)).T.tolist()
    df2 = df1.iloc[yDataRemainSeq,1:]
    if df2.shape[0] == 0:
        return

    outfname = unicode('../DataProcess5/', 'utf-8') + FileName
    df2.to_csv(outfname)

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), ("time","numpy","pandas","scipy.interpolate")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ----- #
# part6 #
# ----- #
# 作图。
# 可能是因为连接服务器用ssh协议，不能直接调用作图函数输出图片，必须附加属性matplotlib.use('Agg') 
# 不能用多进程并行，因为调用matplotlib不是默认方式。没找到解决办法。
matplotlib.use('Agg') 

# 把作图数据的数量上限设定为10000000，否则在数据量过大的时候会报错
matplotlib.rcParams['agg.path.chunksize'] = 10000000

# 把第4步处理后的数据保存到ProcessedData4_3文件夹中
if os.path.exists(u'../DataProcess6'):
    shutil.rmtree(u'../DataProcess6')
    os.mkdir(u'../DataProcess6')
else:
    os.mkdir(u'../DataProcess6')

# 源数据在ProcessedData3_2文件夹中
# inpath = '../DataProcess5'
# uipath = unicode(inpath,'utf-8')
uipath = u'../DataProcess5'

pathfilelist = os.listdir(uipath)

# def ProcessFile(FileName):
# count = 0
for FileName in pathfilelist:
    # infname = unicode('../DataProcess5', 'utf-8') \
    # + u'/' + FileName
    infname = u'../DataProcess5/' + FileName
    df1 = pandas.read_csv(infname)
    if df1.shape[0] == 0:
        # return
        continue
    tData = df1.iloc[:,2].tolist()
    yData = df1.iloc[:,3].tolist()
    time1 = []
    for itime in tData:
        # time2 = datetime.datetime.utcfromtimestamp(itime)
        time2 = datetime.datetime.strptime(itime,'%Y-%m-%d %H:%M:%S')
        time1.append(time2)
    # tData = time1
    tData = matplotlib.dates.date2num(time1)

    # count = count + 1
    f1 = pylab.figure()
    # 创建figure时候如果不传入数字，则自动创建新的figure。否则继续在原figure上面作图
    # plot的输入数据必须是list/array，ndarray不可以
    pylab.plot_date(tData,yData,'-')
    pylab.xlabel('Time')
    pylab.ylabel('Measurement')
    outfname = unicode('../DataProcess6', 'utf-8') \
    + u'/' + FileName[:-4] + '.png'
    # plot1没有savefig属性
    f1.savefig(outfname)
    # 关闭fig
    # fig有clear属性
    # f1.clear()
    pylab.close('all')


# ----- #
# part8 #
# ----- #
# 剔除异常数据之后，做0方差判定、白噪声判定。
if os.path.exists(u'../DataProcess7'):
    shutil.rmtree(u'../DataProcess7')
    os.mkdir(u'../DataProcess7')
else:
    os.mkdir(u'../DataProcess7')

inpath = '../DataProcess5'
uipath = unicode(inpath,'utf-8')

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    infname = unicode('../DataProcess5/', 'utf-8') + FileName
    df1 = pandas.read_csv(infname).iloc[:,1:]
    df1.columns = ['time','data']
    ts1 = df1['data']
    f = open(unicode('../DataProcess7/','utf-8') + FileName[:-4] + unicode('.txt'),'a+')
    if ts1.shape[0] == 0:
        # f = open(unicode('../DataProcess7/','utf-8') + FileName[:-4] + unicode('.txt'),'a+')
        f.write(infname + unicode('长度是0，不进一步分析','utf-8') + u'\n')
        return
    # f = open(unicode('../DataProcess7/Result1.txt','utf-8'),'a+')

    if ts1.var() == 0:        
        f.write(infname + unicode('方差是0，不进一步分析','utf-8') + u'\n')
        return

    try:
        a = statsmodels.stats.diagnostic.acorr_ljungbox(ts1,lags=1)
        if a[1][0] > 0.05:
            f.write(infname + unicode('可能是白噪声，不进一步分析','utf-8') + u'\n')
            return
    except:
        print infname + unicode('白噪声判定失败','utf-8')
        f.write(infname + unicode('白噪声判定失败','utf-8') + u'\n')
        outfname = unicode('../DataProcess7/', 'utf-8') + FileName
        df1.to_csv(outfname)
        return


    f.write(infname + unicode('非白噪声且方差非零','utf-8') + u'\n')
    outfname = unicode('../DataProcess7/', 'utf-8') + FileName
    df1.to_csv(outfname)
    f.close()

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), ("time","numpy","pandas","statsmodels.stats.diagnostic")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ----- #
# part9 #
# ----- #
# 把时间字符串转化成时间戳。
if os.path.exists(u'../DataProcess8'):
    shutil.rmtree(u'../DataProcess8')
    os.mkdir(u'../DataProcess8')
else:
    os.mkdir(u'../DataProcess8')

inpath = '../DataProcess5'
uipath = unicode(inpath,'utf-8')

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    infname = unicode('../DataProcess5/', 'utf-8') + FileName

    df1 = pandas.read_csv(infname)
    ts1 = df1.iloc[:,2]
    tData = df1.iloc[:,1]
    time1 = []
    for itime in tData:
        # time2 = datetime.datetime.utcfromtimestamp(itime)
        time2 = time.mktime(time.strptime(itime,'%Y-%m-%d %H:%M:%S'))
        # time2 = datetime.datetime.strptime(itime,'%Y-%m-%d %H:%M:%S')
        time1.append(time2)

    time1 = pandas.Series(time1)
    # axis=1时，列拼接。默认，行拼接
    df2 = pandas.concat([time1,ts1],axis = 1)
    df2.columns = ['time','data']

    outfname = unicode('../DataProcess8/', 'utf-8') + FileName
    df2.to_csv(outfname)

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), ("time","numpy","pandas","statsmodels.stats.diagnostic")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ------ #
# part10 #
# ------ #
# 数据插值。
# 做cubic样条插值的效果不好，还是做线性插值。
if os.path.exists(u'../DataProcess9'):
    shutil.rmtree(u'../DataProcess9')
    os.mkdir(u'../DataProcess9')
else:
    os.mkdir(u'../DataProcess9')

inpath = '../DataProcess5'
uipath = unicode(inpath,'utf-8')

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    infname = unicode('../DataProcess5', 'utf-8') \
    + u'/' + FileName
    df1 = pandas.read_csv(infname)
    if df1.shape[0] == 0:
        return
        # continue
    tData = df1.iloc[:,1].tolist()
    yData = df1.iloc[:,2].tolist()
    time1 = []
    for itime in tData:
        # 时间戳
        time2 = time.mktime(time.strptime(itime,'%Y-%m-%d %H:%M:%S'))
        time1.append(time2)

    tData = time1
    # 时间序列处理步骤1，去除重复的时间戳。
    # 这一步很关键
    # DataFrame有识别非重复元素的位置的方法！！！
    tDataUnique = numpy.unique(tData)
    # 如果只有一个时间点，不予分析
    if len(tDataUnique) <= 1:
        return

    tDataUniqueSeq = numpy.array(numpy.where(df1.iloc[:,1].duplicated() \
        == False)).T
    
    # numpy里面，列矩阵和列向量的概念是不一样的。
    # shape是[100,1]是矩阵，shape是[100,]的是列向量
    # 列向量索引元素用a[99]，矩阵索引元素用a[99,0]
    tData = numpy.array(tDataUnique)
    yData = numpy.array(numpy.array(yData)[tDataUniqueSeq])[:,0]

    tmin = min(tData)
    tmax = max(tData)
    # 新时间戳
    # tDataNew = numpy.linspace(tmin,tmax,tmax - tmin + 1)
    # 电脑性能有限，数据量太大的话，单位根检验、arima模型内存会溢出
    tDataNew = numpy.linspace(tmin,tmax,40000)
    # 插值的输入只能是ndarry，列矩阵
    # # 三次样条
    # if len(tDataUnique) >= 4:
    #     f = scipy.interpolate.interp1d(tData,yData,kind="cubic")
    # # 二次样条
    # elif len(tDataUnique) == 3:
    #     f = scipy.interpolate.interp1d(tData,yData,kind="quadratic")
    # # 线性
    # else:
    #     f = scipy.interpolate.interp1d(tData,yData,kind="slinear")
    f = scipy.interpolate.interp1d(tData,yData,kind="slinear")
    yDataNew = f(tDataNew)

    df2 = pandas.DataFrame(numpy.column_stack([numpy.array(tDataNew),numpy.array(yDataNew)]))
    outfname = unicode('../DataProcess9/', 'utf-8') + FileName
    df2.to_csv(outfname)

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), ("time","numpy","pandas","scipy.interpolate")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ------ #
# part11 #
# ------ #
# 时间序列的arima处理。
# 但时间序列的数据量太大，效果不好。
if os.path.exists(u'../DataProcess10'):
    shutil.rmtree(u'../DataProcess10')
    os.mkdir(u'../DataProcess10')
else:
    os.mkdir(u'../DataProcess10')

# 源数据在ProcessedData5.1文件夹中
# inpath = '../DataProcess9'
# uipath = unicode(inpath,'utf-8')
uipath = u'../DataProcess9'

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    # infname = unicode('../DataProcess9', 'utf-8') \
    # + u'/' + FileName
    # 函数里面转化成utf-8编码，只能这么处理
    infname = unicode('../DataProcess9/','utf-8') + FileName
    df1 = pandas.read_csv(infname)
    ts1 = df1.iloc[:,2]
    # 方差0的时间序列，认定为稳定序列，而且不能建立ARIMA模型
    if ts1.var() == 0:
        print '方差为0，没有合适的ARIMA模型'
        f = open(unicode('../DataProcess10/result1.txt','utf-8'),'a+')
        f.write(infname + '\n')
        f.write('方差为0，没有合适的ARIMA模型' + '\n\n')
        f.close()
        return
    
    ts2 = ts1
    countdiff = 0
    while 1:
        # adfuller的输入必须是列向量、series。矩阵就算是1行或1列的也不行。
        dftest = statsmodels.tsa.stattools.adfuller(ts2,autolag = 'AIC')
        #dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
        # dfoutput = pandas.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        # for key,value in dftest[4].items():
        #     dfoutput['Critical value (%s)' %key] = value
        if dftest[0] < dftest[4].get('5%'):
            break
        countdiff = countdiff + 1
        ts2 = ts2.diff()

    pmax = 10
    qmax = 10
    bic_matrix=[] #bic矩阵
    for p in range(pmax+1):
        tmp=[]
        for q in range(qmax+1):
            try: #存在部分报错，所以用try来跳过报错。
                tmp.append(statsmodels.tsa.arima_model.ARIMA(ts1,(p,countdiff,q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    if bic_matrix.shape == (0,0):
        p,q = [nan,nan]
        print '没有合适的ARIMA模型'
        f = open(unicode('../DataProcess10/result1.txt','utf-8'),'a+')
        f.write(infname + '\n')
        f.write('没有合适的ARIMA模型' + '\n\n')
        f.close()
    else:
        bic_matrix=pandas.DataFrame(bic_matrix) #从中可找出最小值
        p,q=bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小位置。
        print '最小的p值和q值为: %s、%s'%(p,q)
        model=statsmodels.tsa.arima_model.ARIMA(ts1,(p,countdiff,q)).fit() #建立ARIMA(p,d,q)模型
        model.summary() #给出一份模型报告result:
        f = open(unicode('../DataProcess10/result1.txt','utf-8'),'a+')
        f.write(infname + '\n')
        f.write('最小的p值和q值为: %s、%s'%(p,q) + '\n\n')
        f.close()

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), \
    ("statsmodels.tsa.stattools","numpy","pandas","statsmodels.tsa.arima_model")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ------ #
# part12 #
# ------ #
# 画直方图，看数据的统计分布。
# 非均一值和非白噪声（？）。
if os.path.exists(u'../DataProcess11'):
    shutil.rmtree(u'../DataProcess11')
    os.mkdir(u'../DataProcess11')
else:
    os.mkdir(u'../DataProcess11')

# 源数据在ProcessedData5.1文件夹中
# inpath = '../DataProcess9'
# uipath = unicode(inpath,'utf-8')
uipath = u'../DataProcess9'

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    # infname = unicode('../DataProcess9', 'utf-8') \
    # + u'/' + FileName
    # 函数里面转化成utf-8编码，只能这么处理
    infname = unicode('../DataProcess9/','utf-8') + FileName
    df1 = pandas.read_csv(infname)
    ts1 = df1['data']
    f1 = matplotlib.pyplot.figure('hist')
    # matplotlib.pyplot.xlabel('')
    n,bins,patches = matplotlib.pyplot.hist(ts1,bins=256)
    outfname = unicode('../DataProcess11/', 'utf-8') + FileName[:-4] + '.png'
    f1.savefig(outfname)
    matplotlib.pyplot.close('all')

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), \
    ("numpy","pandas","matplotlib.pyplot")) for i_filename in pathfilelist]
for job in jobs:
    job()
    # print ("Task", i_filename, "is finished")

# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ------ #
# part13 #
# ------ #
# 求时间序列的均值和方差。
# 包括均一序列和白噪声（？）。
if os.path.exists(u'../DataProcess12'):
    shutil.rmtree(u'../DataProcess12')
    os.mkdir(u'../DataProcess12')
else:
    os.mkdir(u'../DataProcess12')

# 源数据在ProcessedData5.1文件夹中
# inpath = '../DataProcess9'
# uipath = unicode(inpath,'utf-8')
uipath = u'../DataProcess9'

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    # infname = unicode('../DataProcess9', 'utf-8') \
    # + u'/' + FileName
    # 函数里面转化成utf-8编码，只能这么处理
    infname = unicode('../DataProcess9/','utf-8') + FileName
    df1 = pandas.read_csv(infname)
    ts1 = df1.iloc[:,2]
    varts1 = ts1.var()
    meants1 = ts1.mean()
    return [varts1,meants1,FileName]

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

ResultAll = []
# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), \
    ("numpy","pandas")) for i_filename in pathfilelist]
for job in jobs:
    Result = job()
    ResultAll.append(Result)
    # print ("Task", i_filename, "is finished")

ResultAll = pandas.DataFrame(ResultAll)
ResultAll.columns=['var','mean','file']
ResultAll.to_csv(u'../DataProcess12/ResultAll.csv')
# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# ------ #
# part14 #
# ------ #
# 求时间序列的趋势，线性拟合。包括均一序列和白噪声。
if os.path.exists(u'../DataProcess13'):
    shutil.rmtree(u'../DataProcess13')
    os.mkdir(u'../DataProcess13')
else:
    os.mkdir(u'../DataProcess13')

# 源数据在ProcessedData5.1文件夹中
# inpath = '../DataProcess9'
# uipath = unicode(inpath,'utf-8')
uipath = u'../DataProcess9'

pathfilelist = os.listdir(uipath)

def ProcessFile(FileName):
# for FileName in pathfilelist:
    # infname = unicode('../DataProcess9', 'utf-8') \
    # + u'/' + FileName
    # 函数里面转化成utf-8编码，只能这么处理
    infname = unicode('../DataProcess9/','utf-8') + FileName
    df1 = pandas.read_csv(infname)
    ts1 = df1.iloc[:,2]
    tData = df1.iloc[:,1]
    time1 = []
    for itime in tData:
        # time2 = datetime.datetime.utcfromtimestamp(itime)
        time2 = time.mktime(time.strptime(itime,'%Y-%m-%d %H:%M:%S'))
        # time2 = datetime.datetime.strptime(itime,'%Y-%m-%d %H:%M:%S')
        time1.append(time2)
    
    ls = gatspy.periodic.LombScargleFast().fit(time1, ts1)

    clf = sklearn.linear_model.LinearRegression()
    clf.fit(numpy.array([time1]).transpose(),numpy.array(ts1).transpose())
    coef1 = clf.coef_[0]
    intercept1 = clf.intercept_
    return [coef1,intercept1,FileName]

# tuple of all parallel python servers to connect with
ppservers = ()
# ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print ("Starting pp with", job_server.get_ncpus(), "workers")

# start_time = time.time()

ResultAll = []
# The following submits jobs and then retrieves the results
jobs = [job_server.submit(ProcessFile,(i_filename,), (), \
    ("numpy","pandas","time","sklearn.linear_model")) for i_filename in pathfilelist]
for job in jobs:
    Result = job()
    ResultAll.append(Result)
    # print ("Task", i_filename, "is finished")

ResultAll = pandas.DataFrame(ResultAll)
ResultAll.columns=['coef','intercept','file']
ResultAll.to_csv(u'../DataProcess13/ResultAll.csv')
# print ("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()