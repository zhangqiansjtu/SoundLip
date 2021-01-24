import numpy as np
import Utils
import os
import re
import time

class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        alldata, self.Xlen, self.alllabel, self.flen, self.allindex, self.alluser, self.len_rate = self.loadfile(filepath)
        alldata, _, self.length = self.cnn_padding1(alldata, self.Xlen,self.flen )
        trainindex, testindex = self.indexsplit(alldata.shape[0], False)
        print(len(trainindex))
        print(len(testindex))
        # self.npfiles = np.array(self.files)
        self.traindata = alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testdata = alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]
        # self.testfile = self.npfiles[testindex]
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.9)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex = []
            for i in range(indexlength):
                if self.allindex[i] < 6000 and self.len_rate[i] >= 0:

                    if self.alluser[i] == 0 and self.len_rate[i] == 0 and self.allindex[i] < 2000:
                        testindex.append(i)
                    elif self.alluser[i] != 0 and self.allindex[i] >= 0:
                            trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
            labelindex = np.array(self.alllabel)
            test_ind = labelindex[testindex]
            test_ind = sorted(range(len(test_ind)), key=lambda k: test_ind[k])
            testindex = np.array(testindex)
            testindex = testindex[test_ind]
            testindex = testindex.tolist()
            # trainindex = []
            # testindex = []
            # for i in range(indexlength):
            #     if self.allindex[i] < 2000 and self.len_rate[i] >= 0:
            #
            #         if self.allindex[i] >= 850 and self.allindex[i] < 880 and self.len_rate[i] == 0:
            #             testindex.append(i)
            #         elif ((self.allindex[i] < 820 or self.allindex[i] >= 880) and self.allindex[i]%2 == 0) or (self.allindex[i] < 850 and  self.allindex[i] >= 820):
            #                 trainindex.append(i)
            # np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self,filepath):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        len_rate = []
        starttime = time.time()
        lasttime = time.time()
        kk = 0
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if (len(res) == 3 and (int(res[2]) >= 0) and int(res[1]) < 4000) or (len(res) == 4 and int(res[1]) < 2000 and int(res[3]) == 0):
                filename = filepath+file
                data = np.load(filename)
                sample = data['datapre']
                sample = sample.astype(np.float32)
                sample1 = sample[:, 0:8]
                samplediff1 = np.diff(sample1,axis=0)*5
                samplepad = np.array([0]*8)
                samplediff1 = np.vstack((samplediff1, samplepad))
                sample1 = np.hstack((sample1, samplediff1))
                sample2 = sample[:, 8:16]
                samplediff2 = np.diff(sample2, axis=0)*5
                samplepad = np.array([0]*8)
                samplediff2 = np.vstack((samplediff2,samplepad))
                sample2 = np.hstack((sample2, samplediff2))
                sample = np.hstack((sample1, sample2))
                # samplenew = []
                # for i in range(sample.shape[1]):
                #     datachannel = sample[:, i]
                #     vmax = np.max(datachannel)
                #     vmin = np.min(datachannel)
                #     datachannel = datachannel / (vmax - vmin)
                #     if i == 0:
                #         samplenew = datachannel
                #     else:
                #         samplenew = np.vstack((samplenew, datachannel))
                # sample = np.transpose(samplenew)
                # sample = sample1
                featurelen = sample.shape[1]
                raw_data.append(sample)
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
                if len(res) == 3:
                    len_rate.append(0)
                else:
                    len_rate.append(int(res[3]))
                kk = kk+1
                if kk % 1000 == 0:
                    nowtime = time.time()
                    print("%d, %0fs" % (kk, nowtime-starttime))


        raw_data = np.array(raw_data)
        raw_label = np.array(raw_label)
        raw_index = np.array(raw_index)
        raw_user = np.array(raw_user)
        return raw_data, raw_data_len, raw_label,featurelen, raw_index, raw_user, len_rate

    def cnn_padding(self, data, slen,flen):
        raw_data = data
        lengths = slen
        median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], median_length)[1] for x in np.array(sample).transpose()]
            padding_data[idx, :, :] = np.array(temp).transpose()
        return padding_data, np.array(slen), median_length

    def cnn_padding1(self, data, slen,flen):
        raw_data = data
        lengths = slen
        median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = np.zeros([flen, median_length])
            sample = np.transpose(sample)
            if slen[idx] < median_length:
                len_diff = median_length - slen[idx]
                len_diff1 = len_diff//2
                len_diff2 = len_diff-len_diff1
                for xidx, x in enumerate(sample):
                    aa = [x[0]]*len_diff1
                    bb = [x[-1]]*len_diff2
                    cc = x.tolist()
                    temp[xidx,:] = [x[0]]*len_diff1+x.tolist()+[x[-1]]*len_diff2
            if slen[idx] > median_length:
                len_diff = slen[idx] - median_length
                # len_diff1 = len_diff//2
                # len_diff2 = len_diff-len_diff1
                temp = sample[:, len_diff:slen[idx]]
            if slen[idx] == median_length:
                temp = sample
            padding_data[idx, :, :] = np.array(temp).transpose()
        return padding_data, np.array(slen), median_length
