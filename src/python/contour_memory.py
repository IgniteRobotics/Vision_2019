import numpy as np
import cv2
import math
import imutils
import itertools

class ContourMemory:
    # count the times you see it! age them out!
    #everything I remember.
    all_memory = []

    # maximum distance and area to use for comparison
    MAX_DISTANCE = 100
    MAX_AREA_DIFF = 100

    #number of consecutive hits to be considered a good contour
    MIN_MATCHES = 3

    #number of consecutive misses before a contour is forgotten
    MAX_MISSES = 2

    def __init__(self, max_dist_diff=100, max_area_diff=100, min_match=3, max_misses=2):
        self.MAX_DISTANCE = max_dist_diff
        self.MAX_AREA_DIFF = max_area_diff
        self.MIN_MATCHES = min_match
        self.MAX_MISSES = max_misses


    '''
    this interleaves digits of number pairs into new numbers 
    which causes numerical distance to coorelate to physical distance
    similar to: https://plus.codes/
    '''
    def weave_pts(self, x, y): 
        #print ('weaving', x, y)
        #zero left pad to equal length
        s1 = str(int(x))
        s2 = str(int(y))
        n = max((len(s1),len(s2)))
        s1 = s1.zfill(n)
        s2 = s2.zfill(n)
        res = "".join(i + j for i, j in itertools.zip_longest(s1, s2,fillvalue='0'))
        #print ('result', res)
        return int(res)

    def age_out(self):
        #print('-- aging contours --')
        keep = []
        for i in range(len(self.all_memory)):
            (loc, area, hits, misses, c) = self.all_memory[i]
            #print('c:',loc, area, hits, misses, c)
            misses += 1
            if misses <= self.MAX_MISSES:
                #print('keep')
                keep.append((loc, area, hits, misses, c))
        self.all_memory = keep


    def evaluate(self, contour):
        # get top left, w,h 
        # calculate the fingerprint
        ((x,y), (w,h), theta) = cv2.minAreaRect(contour)
        #(x,y,h,w) = cv2.minAreaRect(contour)
        #print('eval',x,y,h,w)

        loc_code  = self.weave_pts(x,y)
        new_area = h * w

        matched = False
        for i in range(len(self.all_memory)):
            (loc, area, hits, misses, c) = self.all_memory[i] 
            if abs(loc - loc_code) <= self.MAX_DISTANCE and abs(area - new_area) <= self.MAX_AREA_DIFF: #found it.
                #print('Values:', loc_code, new_area, 'Matched:',loc, area, hits, misses)
                hits += 1
                misses = -1 ##haaack
                self.all_memory[i] = (loc_code, new_area, hits, misses, contour)
                matched = True
                # is breaking out a good idea?
                # is it possible to match more than 1 time?
                if hits >= self.MIN_MATCHES:
                    return True
                else:
                    return False
        if not matched: #it's new
            #print('new contour')
            self.all_memory.append((loc_code, new_area, 1, -1, contour)) #adding -1 misses here is a hack :)
            return False

    def process_contours(self, contours):
        #print('Processing (', len(contours),') contours.')
        good = []
        for c in contours:
            if self.evaluate(c):
                good.append(c)
        self.age_out #immediately age everything, that's why the -1's
        #print(len(good), 'good contours')
        return good