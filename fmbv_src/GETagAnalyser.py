# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:31:06 2013

GE Ultrasound Tag Analyser

@author: gordon
"""

import struct, sys, re


class GETagAnalyser(object):
    def __init__(self, fname):
        self.m_fname = None
        if fname is not None:
            self.m_fname = fname
        self.m_tagdict = []

    def readAllTags(self):
        with open(self.m_fname, 'rb') as f:
            hdr = f.read(16)
            print(hdr)
            # find end
            f.seek(0, 2)
            fend = f.tell()
            print(fend)
            # back to start
            f.seek(16, 0)
            while f.tell() <= fend - 4:
                t1, t2, s, tag_data, f = self.readNextTag(f)
                if s < 16:
                    print(hex(t1), hex(t2), tag_data)
                else:
                    print(hex(t1), hex(t2), s)

    def getDataFromTags(self, tag1, tag2=None):
        if isinstance(tag1, str) or isinstance(tag2, str):
            print("DataType needs to be hex or int!")

        with open(self.m_fname, 'rb') as f:
            hdr = f.read(16)
            # find end
            f.seek(0, 2)
            fend = f.tell()
            # back to start
            f.seek(16, 0)
            if tag2 is None:
                return_data = list()
            else:
                return_data = None
            while f.tell() <= fend - 4:
                t1, t2, s, tag_data, f = self.readNextTag(f)
                if (t1 == tag1) and (t2 == tag2):
                    return_data = tag_data
                    break
                if t1 == tag1:
                    if t2 is None:
                        return_data.append(tag_data)

            return return_data

    def getTagsFromData(self, test_data, contains=False):
        with open(self.m_fname, 'rb') as f:
            hdr = f.read(16)
            # find end
            f.seek(0, 2)
            fend = f.tell()
            f.seek(16, 0)
            tag_1, tag_2 = None, None
            while f.tell() <= fend - 4:
                t1, t2, s, tag_data, f = self.readNextTag(f)
                tag_data = re.sub(r'\W+', '', str(tag_data))
                if contains:
                    if tag_data.find(test_data) > 0:
                        tag_1 = hex(t1)
                        tag_2 = hex(t2)
                        break
                else:
                    if tag_data == test_data:
                        tag_1 = hex(t1)
                        tag_2 = hex(t2)
                        break

            return tag_1, tag_2

    def readNextTag(self, f):
        firsttag = f.read(2)
        firsttag = struct.unpack('@H', firsttag)

        secondtag = f.read(2)
        secondtag = struct.unpack('@H', secondtag)

        size = f.read(4)
        size = struct.unpack('@I', size)
        data = f.read(size[0])

        tag1, tag2, size, data = firsttag[0], secondtag[0], size[0], data
        return tag1, tag2, size, data, f


    def compareFiles(self,second_file):
        differences = list()

        with open(second_file, 'rb') as f2:
            hdr2 = f2.read(16)
            f2.seek(0, 2) # find end
            fend2 = f2.tell()
            f2.seek(16, 0) # back to start

            with open(self.m_fname, 'rb') as f:
                hdr = f.read(16)
                f.seek(0, 2) # find end
                fend = f.tell()
                f.seek(16, 0) # back to start
                fend_end = min(fend ,fend2)
                while f.tell() <= fend_end - 4:
                    t1, t2, s, tag_data, f = self.readNextTag(f)
                    t1_f2, t2_f2, s_f2, tag_data_f2, f2 = self.readNextTag(f2)
                    if tag_data != tag_data_f2 and t2 != 0xffff:
                        differences.append((t1,t2))

        return differences



    def writeToDCMFormat(self, output_path):

        with open(output_path, 'wb') as fout:
            fout.write(bytearray(128))
            fout.write('DICM')
            with open(self.m_fname, 'rb') as f:
                hdr = f.read(16)
                # find end
                f.seek(0, 2)
                fend = f.tell()
                # back to start
                f.seek(16, 0)
                vol_data = f.read(fend - 4)
                fout.write(vol_data)


