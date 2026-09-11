#!/usr/bin/python
# -*- coding: utf-8 -*-
#########################################################
## Minimum modules for g2p, th2ipa, th2roman, word_segment
## Thai Language Toolkit : version  1.3.8
## word_segmentation, syl_segementation written by Wirote Aroonmanakun
## Implemented :
##      chunk, ner_tag, segment, word_segment, syl_segment, word_segment_mm, word_segment_nbest,
##      g2p, th2ipa, th2roman, spell_variants, pos_tag, pos_tag_wordlist, 
##      read_thaidict, reset_thaidict, check_thaidict
##      spell_candidates,
##
## Copyright (c) 2018 Dept of Linguistics, Chulalongkorn University. All rights reserved.

## Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

## 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
## 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
## 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#########################################################

import re
import math
import os
from copy import deepcopy
from collections import defaultdict

#import pkg_resources
import pickle

# Pre-compiled regex cache: pattern string -> compiled regex object
_RE_CACHE = {}

##########################################################
## Read Dictionary in a text format one word per one line
##########################################################
def read_thaidict(Filename):
    global TDICT
#    ATA_PATH = pkg_resources.resource_filename('tltk', '/')

    if not os.path.exists(Filename):
        path = os.path.abspath(__file__)
        ATA_PATH = os.path.dirname(path)
        Filename = ATA_PATH + '/' + Filename
    file1 = open(Filename, 'r', encoding ='cp874')
    for line in  file1:
        w = line.rstrip()
        w = re.sub(r'\.','\\\.',w)
        TDICT[w] = 1
    return(1)

def read_thdict(Filename):
    global TDICT
    fileObject = open(Filename,'rb')  
    TDICT = pickle.load(fileObject)


##########################################################
## Clear Dictionary in a text format one word per one line
##########################################################
def reset_thaidict():
    global TDICT
    TDICT.clear()
    return(1)

#### Check whether the word existed in the dictionary 
def check_thaidict(Word):
    global TDICT
    if Word in TDICT:
        return(1)
    else:
        return(0)
    
    


   

#############################################################################################################
### Thai grapheme to phoneme
### Input = a chunk of Thai texts
### orginal written in Perl, ported to Python on May 17, 2018
#############################################################################################################
def g2p(Input):
    global SegSep
    global SSegSep
    output = ""
    out = ""
    
    Input = preprocess(Input)
    sentLst = Input.split(SegSep)
    for s in sentLst:
        inLst = s.split(SSegSep)
        for inp in inLst:
            if inp == '': continue            
            objMatch = re.match(r"[^ก-์]+",inp)
            if objMatch:
                out = inp+'<tr/>'+inp
            else:
                y = sylparse(inp)
                out = wordparse(y)
            output = output+out+WordSep
        output = output+'<s/>'    ####write <s/> output for SegSep   
    return(output)        

## return all transcriptions based on syllable parse
def g2p_all(inp):
    output = []
    NORMALIZE_IPA = [ ('O', '\u1D10'), ('x', '\u025B'), ('@', '\u0264'), ('N', '\u014B'), ('?', '\u0294'),('U','\u026F'),('|',' '),('~','.'),('^','.'),("'",'.'),('4','5'), ('3','4'), ('2','3'), ('1','2'), ('0','1')]
    
    if inp == '': return([])            
    objMatch = re.match(r"[^ก-์]+$",inp)
    if objMatch:
        output = [(inp,inp)]
    else:
        lst = sylparse_all(inp)
        if lst == []: return([('','')])
        for (th,tran) in lst:
            tran = re.sub(r"([aeiouUxO@])\1",r"\1ː",tran)
            tran = re.sub(r"([ptkc])h",r"\1ʰ",tran)
            for k, v in NORMALIZE_IPA:
                tran = tran.replace(k, v)
            output.append((th,tran))
#            print(th,tran)
    return(output)        


#############################################################################################################
####### Segment syllable using trigram statistics, only strings matched with a defined syllable pattern will be created
###### all pronunciations of each syllable
def sylparse(Input):
    global SylSep
    global PRON
    global PRONUN
    
    PRONUN = defaultdict(list)
    schart = defaultdict(dict)
    probEnd = defaultdict(float)
    schartx = {}
    schart.clear()
    probEnd.clear()
    tmp = []
    
    EndOfInput = len(Input)
    for f in PRON:
        compiled_f = _RE_CACHE.get(f)
        if compiled_f is None:
            compiled_f = _RE_CACHE[f] = re.compile(f)
        for i in range(EndOfInput):
            Inx = Input[i:]
            matchObj = compiled_f.match(Inx)
            if matchObj:
                keymatch = matchObj.group()
                try:
                    matchObj.group(3)
                    charmatch = matchObj.group(1) + ' ' + matchObj.group(2) + ' ' + matchObj.group(3)
                except IndexError:
                    try:
                        matchObj.group(2)
                        charmatch = matchObj.group(1) + ' ' + matchObj.group(2) 
                    except IndexError:
                        try:
                            matchObj.group(1)
                            charmatch = matchObj.group(1) 
                        except IndexError:
#                            PRONUN[matchObj.group()].append(PRON[f])
#                            print("ADD",PRON[f])
                            PRONUN[matchObj.group()].extend(PRON[f])
                k=i+len(matchObj.group())
                schart[i][k] = [matchObj.group()]
#                print("Match PRON",schart[i][k],i,k)
                #### expand all pronunciations
                for PronF in PRON[f]:
#                    print(f,PronF)
                    codematch = re.sub(r"[^AKYDZCRX]","",PronF)
                    if codematch:
#                        print("code char",codematch,charmatch)            
                        phone = ReplaceSnd(PronF,codematch,charmatch)
#                        print("phone",phone)
                        if  NotExceptionSyl(codematch,charmatch,keymatch,phone):
                            (phone,tone) = ToneAssign(keymatch,phone,codematch,charmatch)
 #                           print('assign tone',tone,' to',keymatch)
                            if (tone < '5'): phone = re.sub(r'8',tone,phone)          
                            (keymatch,phone) = TransformSyl(keymatch,phone)         
                        PRONUN[''.join(schart[i][k])].append(phone)
 #                       print("Add",PronF,''.join(schart[i][k]), phone)
                        if  re.match(r'ทร',keymatch)  and  re.match(r"thr",phone):            #### gen more syllable  ทร   thr => s
                            phone=re.sub(r"thr","s",phone) 
                            PRONUN[''.join(schart[i][k])].append(phone)
#                            print("Add2",PronF,''.join(schart[i][k]), phone)
                        probEnd[(i,k)] = prob_trisyl(schart[i][k])

#                codematch = PRON[f]
#                codematch = re.sub(r"[^AKYDZCRX]","",codematch)
#                if codematch:
#                    print("code char",codematch,charmatch)            
#                    phone = ReplaceSnd(PRON[f],codematch,charmatch)
#                    if  NotExceptionSyl(codematch,charmatch,keymatch,phone):
#                        (phone,tone) = ToneAssign(keymatch,phone,codematch,charmatch)
#                        print('assign tone',tone,' to',keymatch)
#                        if (tone < '5'): phone = re.sub(r'8',tone,phone)          
#                        (keymatch,phone) = TransformSyl(keymatch,phone)         
#                    PRONUN[''.join(schart[i][k])].append(phone)
#                    print("Add",PRON[f],''.join(schart[i][k]), phone)
#                    if  re.match(r'ทร',keymatch)  and  re.match(r"thr",phone):            #### gen more syllable  ทร   thr => s
#                        phone=re.sub(r"thr","s",phone) 
#                        PRONUN[''.join(schart[i][k])].append(phone)
#                        print("Add2",PRON[f],''.join(schart[i][k]), phone)
#                    probEnd[(i,k)] = prob_trisyl(schart[i][k])


    for j in range(EndOfInput):
        schartx = deepcopy(schart)
        if j in schart[0]:
            s1 = schart[0][j]
            for k in schart[j]:
                    s2 = schart[j][k]
                    tmp = mergekaran1(s1+s2)
                    if k not in schart[0]:                        
                        schartx[0][k] = tmp
                        probEnd[(0,k)] = prob_trisyl(tmp)
#                        print("Not Found K",tmp,probEnd[(0,k)])
                    else:
                        p = prob_trisyl(tmp)
                        if p > probEnd[(0,k)]:
                            schartx[0][k] = tmp 
                            probEnd[(0,k)] = p
#                            print("Found K new",tmp,probEnd[(0,k)])
        schart = deepcopy(schartx)
    if EndOfInput in schart[0]:    
        return(SylSep.join(schart[0][EndOfInput]))
    else:
        return('<Fail>'+Input+'</Fail>')

def sylparse_all(Input):
    global SylSep
    global PRON
    global PRONUN
    
    PRONUN = defaultdict(list)
    phchart = defaultdict(dict)
    schartx = {}
    phchart.clear()
    tmp = []
    
    EndOfInput = len(Input)
    for f in PRON:
        if f == '([ก-ฮ])' and PRON[f] == 'XOO': continue
        compiled_f = _RE_CACHE.get(f)
        if compiled_f is None:
            compiled_f = _RE_CACHE[f] = re.compile(f)
        for i in range(EndOfInput):
            Inx = Input[i:]
            matchObj = compiled_f.match(Inx)
            if matchObj:
                keymatch = matchObj.group()
                try:
                    matchObj.group(3)
                    charmatch = matchObj.group(1) + ' ' + matchObj.group(2) + ' ' + matchObj.group(3)
                except IndexError:
                    try:
                        matchObj.group(2)
                        charmatch = matchObj.group(1) + ' ' + matchObj.group(2) 
                    except IndexError:
                        try:
                            matchObj.group(1)
                            charmatch = matchObj.group(1) 
                        except IndexError:
#                            PRONUN[matchObj.group()].append(PRON[f])
                            PRONUN[matchObj.group()].extend(PRON[f])
                k=i+len(matchObj.group())
                frm = matchObj.group()

                for PronF in PRON[f]:
#                codematch = PRON[f]
                    codematch = re.sub(r"[^AKYDZCRX]","",PronF)
                    if codematch:
#                        print("code char",codematch,charmatch)            
                        phone = ReplaceSnd(PronF,codematch,charmatch)
                        if  NotExceptionSyl(codematch,charmatch,keymatch,phone):
                            (phone,tone) = ToneAssign(keymatch,phone,codematch,charmatch)
                            if (tone < '5'): phone = re.sub(r'8',tone,phone)          
                            (keymatch,phone) = TransformSyl(keymatch,phone)
    #                        phchart[0][k] = {frm+'/'+phone:1}
                            if k not in phchart[i]:
                                phchart[i][k] = {frm+'/'+phone:1}
                            else:
                                phchart[i][k].update({frm+'/'+phone:1})
    #                        print(i,k,frm,phone)     
                            if  re.match(r'ทร',keymatch)  and  re.match(r"thr",phone):            #### gen more syllable  ทร   thr => s
                                phone=re.sub(r"thr","s",phone) 
    #                            PRONUN[''.join(schart[i][k])].append(phone)
                            if k not in phchart[i]:
                                phchart[i][k] = {frm+'/'+phone:1}
                            else:
                                phchart[i][k].update({frm+'/'+phone:1})
    
    for j in range(EndOfInput):
        schartx = deepcopy(phchart)
        if j in phchart[0]:
            for s1 in phchart[0][j]:
                for k in phchart[j]:
                    for s2 in phchart[j][k]:
    #                    tmp = mergekaran1(s1+s2)
                        tmp = s1+'~'+s2
                        if k not in schartx[0]:
                            schartx[0][k] = {tmp:1}
                        else:
                            schartx[0][k].update({tmp:1})
        phchart = deepcopy(schartx)
        
    outlst = []
    if EndOfInput not in phchart[0]: return([])
    for out in phchart[0][EndOfInput]:
        form = []
        ph = []
        for x in out.split('~'):
            (f,p) = x.split('/')
            form.append(f)
            ph.append(p)
        outlst.append(('~'.join(form),'~'.join(ph)))
#        print(form,ph)
    return(outlst)    

def ReplaceSnd(phone,codematch,charmatch):
     global stable
     snd = phone
     tmp1Lst = charmatch.split(' ')   #get character
     i=0
     for x in list(codematch):
          s = stable[x][tmp1Lst[i]]
          snd = re.sub(x,s,snd)
          i += 1 
     snd += '8'
#     print('Sound',snd)
     return(snd)

def NotExceptionSyl(codematch,charmatch,form,phone):
    if re.search(r'\.',form):  return(1)
##  check pronunciation marked in syllable dict, if it exists and it is different from the current phone, disregard current phone.
    if 'CR' in codematch:        
#exception for CR = ถร  ผร  ดล  ตล ถล ทล บล ดว ตว ถว ทว บว ปว ผว สว
        if re.match(r'ผ ร|ด ล|ต ล|ท ล|ด ว|ต ว|ท ว|บ ว|ป ว|พ ว|ฟ ว|ผ ว|ส ล|ส ว|ร ร|ศ ล|ศ ว',charmatch):  return(-1)
#exception for AK = กย กง ขง คง คม จง จน จก ฉย ชง ดย ดง ดน ดม ถย บย บง บน บม ปง ผม พง ฟย ฟง ฟน ฟม ซย ซง ซน ซม  ถร บล บว ปว พร พว นน ยด คว
    if 'AK' in codematch:  #check for leadnng and followinf consinant
        clst = charmatch.split(' ')
        if clst[1] not in AK[clst[0]]: return(-1)

#Case 1 xัว with sound like "..aw"
    if re.search(r'\u0E31[\0E48-\u0E4B]?ว]',form) and 'aw' in phone: return(-1)
#Case 5 check for speller ข Only 3 vowel forms can be used  ัุ   เ
    if re.search(r'[ก-ฮ] ข',charmatch) and not re.search(r'[\u0E38\u0E31\u0E40]',form): return(-1)
# Case  xร - xon   except  Xรน (กรน ปรน)
    if re.search(r'[ก-ฮ] ร$',charmatch) and re.search(r'.an',phone): return(-1)
    return(1)

#######################################
# Tone assign :  ม้าน, maan, codematch XY,  charmatch  ม น,  => return 3
# ToneAssign($keymatch,$phone,$codematch,$charmatch); 
#######################################
def ToneAssign(keymatch,phone,codematch,charmatch):
#    print("ToneAssign:",keymatch,phone,codematch,charmatch)
    if phone == '' : return('','9')
    lead = ''
    init = ''
    final = ''
    if re.search(r'[0-4]8',phone):   # tone is already assigned
        phone = re.sub(r'([0-4])8',r'\1',phone)
        return(phone,'')
    if 'X' in codematch or codematch == 'GH' or codematch == 'EF':
        lx = charmatch.split(' ')
        lead = ''
        init = lx[0]
        if len(lx) > 1:
            final = lx[1]
        else: final = ''    
    elif re.search(r'AK',codematch) or re.search(r'CR',codematch):
#        (lead, init, final) = charmatch.split(' ')
        lx = charmatch.split(' ')
        lead = lx[0]
        if len(lx) > 2:
            final = lx[2]
            init = lx[1]
        elif len(lx) >1:    
            init = lx[1]
            final = ''

    deadsyll = DeadSyl(phone)
#    print('dead syallble',phone,deadsyll,lead,init,final)

### change + for leading syllable
    if "+'" in phone:
#        print('found leading',phone,lead)
        if lead in 'ผฝถขสหฉศษ':
            phone = re.sub(r'\+','1',phone)
        elif lead in 'กจดตบปอ':
            phone = re.sub(r'\+','1',phone)
        else:    
            phone = re.sub(r'\+','3',phone)

#### normal syllable 
    if init in 'กจดตฎฏบปอ':   # middle consonant
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
            else: return(phone,'0')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
            else: return(phone,'1')
    elif init in 'ขฃฉฐถผฝสศษห':   # high consonant
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')  #Maijatawa
            else: return(phone,'4')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')  #Maijatawa
            else: return(phone,'1')
    elif init in 'งญณนมยรลวฬ' and lead != '' and lead in 'ขฃฉฐถผฝสศษห':  #low consonant single
#        if lead in 'ขฃฉฐถผฝสศษห':   # lead by high consonant
            if deadsyll == 'L':
                if re.search(r'\u0E48',keymatch): return(phone,'1')   #Maiaek
                elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
                elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
                elif re.search(r'\u0E4B',keymatch): return(phone,'9')  #Maijatawa
                else: return(phone,'4')
            else:
                if re.search(r'\u0E48',keymatch): return(phone,'9')   #Maiaek
                elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
                elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
                elif re.search(r'\u0E4B',keymatch): return(phone,'9')  #Maijatawa
                else: return(phone,'1')
    elif init in 'งญณนมยรลวฬ' and lead != '' and lead in 'กจดตฎฏบปอ':  #low consonant single
#        elif lead in 'กจดตฎฏบปอ':  #lead by middle consonant
            if deadsyll == 'L':
                if re.search(r'\u0E48',keymatch): return(phone,'1')   #Maiaek
                elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
                elif re.search(r'\u0E4A',keymatch): return(phone,'3')  #Maitri
                elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
                else: return(phone,'0')
            else:
                if re.search(r'\u0E48',keymatch): return(phone,'9')   #Maiaek
                elif re.search(r'\u0E49',keymatch): return(phone,'2')  #Maitoo
                elif re.search(r'\u0E4A',keymatch): return(phone,'3')  #Maitri
                elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
                else: return(phone,'1')
    elif init in 'คฅฆชฌซฑฒทธพภฟฮงญณนมยรลวฬฤฦ': #low consonant
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'2')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'3')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')  #Maijatawa
            else: return(phone,'0')        
        elif re.search(r'[aeiouxOU\@][aeiouxOU\@]+',phone):  # long vowel
            if re.search(r'\u0E48',keymatch): return(phone,'9')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'3')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
            else: return(phone,'2')
        else:    # short vowel
            if re.search(r'\u0E48',keymatch): return(phone,'2')   #Maiaek
            elif re.search(r'\u0E49',keymatch): return(phone,'9')  #Maitoo
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')  #Maitri
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')  #Maijatawa
            else: return(phone,'3')

#########################################
# Check whether it's a dead syllable : input is a pronunciation, return 'D' or 'L'
##########################################
def DeadSyl(phone):
    inx = phone
    inx = re.sub('ch','C',inx)
    inx = re.sub(r'[0-4]','',inx)
    if re.search(r'[mnwjlN]8?$',inx):
        return('L')
    elif re.search(r'[pktfscC]8?$',inx):
        return('D')
    elif re.search(r'([aeiouxOU\@])\1',inx):  # vowel length > 1
        return('L')
    else:
        return('D')
    
def TransformSyl(form,phone):
# xxY[12]  eeY[12] @@Y[12]  => ลดสระสั้น  ใน Y = [nmN]
    if re.search(r'xx[nmN][12]',phone):
        phone = re.sub(r'xx','x',phone)
    elif re.search(r'ee[nmN][12]',phone):
        phone = re.sub(r'ee','e',phone)
    elif re.search(r'\@\@[nmN][12]',phone):
        phone = re.sub(r'\@\@','\@',phone)
#Case 1 อยxxx change sound "?a1'jxxx" to "jxxx"
    if re.search(r'^อย่า$|^อยู่$|^อย่าง$|^อยาก$',form) and "'" in phone:
        x = phone.split("'")
        phone = x[-1]
#Case 2 หxxx change spund "ha1'xxx" to "xxx"
    elif 'ห' in form and 'ha1' in phone and not re.search(r'หนุ$|หก|หท|หพ|หฤ|หโ',form):
        x = phone.split("'")
        phone = x[-1]
#Case 3 arti-cluster sound, sound "r" is deleted
    elif re.search(r'[จซศส]ร',form) and re.search(r'[cs]r',phone) and re.search(r"[^']",phone):
        phone = re.sub('r','',phone)
    return (form,phone)
    
#### word segment and select the most likely pronunciation in a word    
def wordparse(Input):
    global TDICT
    global EndOfSent
    global chart
    global SegSep
    global WordSep
    global CollocSt
    
    maiyamok_find = r"(<tr/>|\|)" + r"([?a-zENOU0-9~'@^]+?)"  + r"[|~]ๆ"
    maiyamok_rep = r"\1\2" + WordSep + r"\2"

    part = []
    chart = defaultdict(dict)
    SylSep = '~'
    outx = ""
    chart.clear()
    CollocSt = defaultdict(float)
    
    part = Input.split(SegSep)
#    print('part',part,'xxx')
    for inx in part:
        SylLst = inx.split(SylSep)
        EndOfSent = len(SylLst)
        ######### Gen unknown word by set each syllable as a potential word
#        gen_unknown_thaiw(SylLst)
        for i in range(EndOfSent):
            chart[i][i+1] = [SylLst[i]]
        ############################################################
        for i in range(EndOfSent):
            for j in range(i,EndOfSent+1):
                wrd = ''.join(SylLst[i:j])
                if wrd in TDICT:
#                    chart[i][j] = [wrd]
                    chart[i][j] = ['~'.join(SylLst[i:j])]
                    if j > i+1:   ### more than one syllable, compute St
                        St = 0.0
                        NoOfSyl = len(SylLst[i:j])
                        for ii in range(i,j-1):
                            St += compute_colloc("mi",SylLst[ii],SylLst[ii+1])
                        CollocSt[(i,j)] = St    #### Compute STrength of the word
                    else:   ### one sylable word St = 0
                        CollocSt[(i,j)] = 0.0
        if chart_parse():
            outx += WordSep.join(chart[0][EndOfSent])
            outx = outx.replace('~ๆ','|ๆ')
            outx += '<tr/>'
            outp = []
            for  wx in chart[0][EndOfSent]:
                tmp = wx.split(SylSep)
                op = SelectPhones(tmp)    
                outp.append(op)
            outx += WordSep.join(outp)
### replace duplicate word for ๆ
            outx = re.sub(maiyamok_find,maiyamok_rep,outx)        
            return(outx)
        else:
            return("<Fail>"+Input+"</Fail>")
    
## input = list of syllables
## output = syl/pron-syl/pron-syl/pron
def SelectPhones(slst):
   global PRONUN 
   p=''
   out = []
   prmax = 0.

   slst = ['|'] + slst + ['|']
#   print('slist',slst)
   i = 1
   for i in range(1,len(slst)-1):
        outp = ''
        prmax = 0.
#        if slst[i] == '|': continue
        if len(PRONUN[slst[i]]) == 1:
            out.append(PRONUN[slst[i]][0])
            continue
        else:
            for p in PRONUN[slst[i]]:
                pr = ProbPhone(p, slst[i-1],slst[i],slst[i+1])
#                print(slst[i],' pronounce ',p,pr,prmax)
                if pr > prmax:
                   prmax = pr
                   outp = p
                elif pr == prmax:
                   if re.search(r"'",p)  and len(p) > len(outp):
                      prmax = pr
                      outp = p
        out.append(outp)
#        print('out',slst[i],out)
        i += 1
#   print('Select Phone',out)       
   return('~'.join(out))


####################
def ProbPhone(p,pw,w,nw):
    global PhSTrigram
    global FrmSTrigram
    global PhSBigram
    global FrmSBigram
    global PhSUnigram
    global FrmSUnigram
    global AbsUnigram
    global AbsFrmSUnigram

    p3=0.
    p2=0.
    p1=0.
    p0=0.
    if PhSTrigram[(pw,w,nw,p)] > 0.:
        p3 = (1. + math.log(PhSTrigram[(pw,w,nw,p)])) / (1. + math.log(FrmSTrigram[(pw,w,nw)]))
#        print('Trigram',PhSTrigram[(pw,w,nw,p)])
    if PhSBigram[(pw,w,p)] > 0.:
#        print('Bigram1',PhSBigram[(pw,w,p)])
        p2 = (1. + math.log(PhSBigram[(pw,w,p)])) / (1. + math.log(FrmSBigram[(pw,w)])) * 0.25
### check w and next w because following syllable is important to determine the linking sound  give it more weigth x 3/4
    if PhSBigram[(w,nw,p)] > 0.:
#        print('Bigram2',PhSBigram[(w,nw,p)])
        p2 = p2 + (1. + math.log(PhSBigram[(w,nw,p)])) / (1. + math.log(FrmSBigram[(w,nw)])) * 0.75
    if PhSUnigram[(w,p)] > 0.:
#        print('Unigram',PhSUnigram[(w,p)])
        p1 = (1 + math.log(PhSUnigram[(w,p)])) / (1. + math.log(FrmSUnigram[w]))

### get abstract form of sounds
    abs_w = re.sub(r"[่้๊๋]","",w)
    abs_w = re.sub(r"[ก-ฮ]","C",abs_w)
    abs_p = re.sub(r"[0-9]","",p)
    abs_p = re.sub(r"[^aeio@OuxU]","C",abs_p)
    if AbsUnigram[(abs_w,abs_p)] > 0.:
        p0 = (1 + math.log(AbsUnigram[(abs_w,abs_p)])) / (1. + math.log(AbsFrmSUnigram[abs_w]))
#        print('AbsUnigram',p0)
    prob =  0.8*p3 + 0.16*p2 + 0.03*p1 + 0.00001*p0 + 0.00000000001
#    prob =  0.8*p3 + 0.16*p2 + 0.03*p1 + 0.00000000001
    return(prob)


    
def th2ipa(txt):
    out = ''
    NORMALIZE_IPA = [ ('O', '\u1D10'), ('x', '\u025B'), ('@', '\u0264'), ('N', '\u014B'), ('?', '\u0294'),('U','\u026F'),('|',' '),('~','.'),('^','.'),("'",'.'),('4','5'), ('3','4'), ('2','3'), ('1','2'), ('0','1')]
    inx = g2p(txt)
    for seg in inx.split('<s/>'):
        if seg == '': continue
        (th, tran) = seg.split('<tr/>')
        tran = re.sub(r"([aeiouUxO@])\1",r"\1ː",tran)
        tran = re.sub(r"([ptkc])h",r"\1ʰ",tran)
        for k, v in NORMALIZE_IPA:
            tran = tran.replace(k, v)
        out += tran+'<s/>'
    return(out)

def th2roman(txt):
    out = ''
    NORMALIZE_ROM = [ ('O', 'o'), ('x', 'ae'), ('@', 'oe'), ('N', 'ng'), ('U','ue'), ('aw','ao'), ('iw','io'), ('ew','eo'), ('?',''), ('|',' '), ('~','-'),('^','-'),("'",'-')]
    inx = g2p(txt)
    for seg in inx.split('<s/>'):
        if seg == '': continue
        (th, tran) = seg.split('<tr/>')
        tran = re.sub(r"([aeiouUxO@])\1",r"\1",tran)
        tran = re.sub(r"[0-9]",r"",tran)
        for k, v in NORMALIZE_ROM:
            tran = tran.replace(k, v)
        tran = re.sub(r"([aeiou])j",r"\1i",tran)
        tran = tran.replace('j','y')
        tran = re.sub(r"c([^h])",r"ch\1",tran)
        tran = re.sub(r"\-([^aeiou])",r"\1",tran)
        out += tran+'<s/>'
    return(out)
    
### end of modules used in g2p  ###############    
##############################################################################################################



###################################################################
###### Thai word segmentation using maximum collocation approach
###### Input is a list of syllables
###### also add each syllable as a potential word
def wordseg_colloc(Input):
    global TDICT
    global EndOfSent
    global chart
    global SegSep
    global WordSep
    global CollocSt
    
    part = []
    chart = defaultdict(dict)
    SylSep = '~'
    outx = ""
    chart.clear()
    CollocSt = defaultdict(float)
    
    part = Input.split(SegSep)
    for inx in part:
        SylLst = syl_segment(inx).split('~')
        if SylLst[-1] == '<s/>': SylLst.pop()
#        SylLst = inx.split(SylSep)
        EndOfSent = len(SylLst)
        ######### Gen unknown word by set each syllable as a potential word
        gen_unknown_w(SylLst)
#        for i in range(EndOfSent):
#            chart[i][i+1] = [SylLst[i]]
        eng_abbr(SylLst)    
        ############################################################
        for i in range(EndOfSent):
            for j in range(i,EndOfSent+1):
                wrd = ''.join(SylLst[i:j])
                if wrd in TDICT:
                    chart[i][j] = [wrd]
                    if j > i+1:   ### more than one syllable, compute St
                        St = 0.0
                        NoOfSyl = len(SylLst[i:j])
                        for ii in range(i,j-1):
                            St += compute_colloc("mi",SylLst[ii],SylLst[ii+1])
#                            print (SylLst[ii],SylLst[ii+1],xx)
                        CollocSt[(i,j)] = St    #### Compute STrength of the word
#                        print(i,j,wrd,CollocSt[(i,j)])
                    else:   ### one sylable word St = 0
                        CollocSt[(i,j)] = 0.0
        if chart_parse():
#            return(chart[0][EndOfSent])
            outx += WordSep.join(chart[0][EndOfSent])
            return(outx)
        else:
            return("<Fail>"+Input+"</Fail>")
        



####################################################################
#### Word segmentation using Maximal Matching (minimal word) approach
#### Input = Thai string,  method = mm|colloc|ngram|w2v , 
####   spellchk=yes|no 
######################################################################
def word_segment(Input,method='colloc',spellchk='no'):
    global SegSep
    global SSegSep
    output = ""
    out = ""
    
    Input = preprocess(Input)
    sentLst = Input.split(SegSep)
    for s in sentLst:
#        print ("s:",s)
        inLst = s.split(SSegSep)
        for inp in inLst:
            if inp == '': continue            
            objMatch = re.match(r"[^ก-์]+",inp)
            if objMatch:
                out = inp
            else:
#                print('ss:',inp)
                if method == 'mm' or method == 'ngram':
                    out = wordseg_mm(inp,method,spellchk)
                elif method == 'colloc':
                    out =wordseg_colloc(inp)
                elif method == 'w2v':
                    out =wordseg_w2v(inp,spellchk)
#                elif method == 'ngram':
#                    out =wordseg_mm(inp,method,spellchk)
            output = output+out+WordSep
#        output = output.rstrip(WordSep)
        output = output+'<s/>'    ####write <s/> output for SegSep   
    return(output)

def word_segment_mm(Input):
    return(word_segment(Input,method='mm'))

def wordseg_mm(Input,method,spellchk):    
    global TDICT
    global EndOfSent
    global chart
    global SegSep
    global WordSep


    part = []
    chart = defaultdict(dict)
    outx = ""
    chart.clear()
    
    part = Input.split(SegSep)
    for inx in part:
        if method == 'ngram':
            SylLst = syl_segment(inx).split('~')
            SylLst.pop()
#            print('syl',SylLst)
        else:
            SylLst = list(inx)
        EndOfSent = len(SylLst)    
        if spellchk == 'yes' and method == 'ngram':            
            gen_unknown_thaiw(SylLst)
        eng_abbr(SylLst)    
        for i in range(EndOfSent):
            for j in range(i,EndOfSent+1):
                wrd = ''.join(SylLst[i:j])
                if wrd in TDICT:
                    chart[i][j] = [wrd]
                    
        if method == 'ngram':            
            if chartparse_ngram():
                outx += WordSep.join(chart[0][EndOfSent])
            else:
                outx += "<Fail>"+Input+"</Fail>"
        elif method == 'mm':        
            if chartparse_mm():
                outx += WordSep.join(chart[0][EndOfSent])
            else:
                outx += "<Fail>"+Input+"</Fail>"
    return(outx)        

#########  Chart Parsing, ceate a new edge from two connected edges, always start from 0 to connect {0-j} + {j+k}
#########  If minimal word appraoch is chosen, the sequence with fewest words will be selected
def chartparse_mm():
    global chart
    global EndOfSent
    
    for j in range(EndOfSent):
        chartx = deepcopy(chart)
        if j in chart[0]:
            s1 = chart[0][j]
            for k in chart[j]:
                    s2 = chart[j][k]
#                    print 0,j,k
                    if k not in chart[0]:                        
                        chartx[0][k] = s1+s2
                    else:
                        if len(s1)+len(s2) <= len(chart[0][k]):
                            chartx[0][k] = s1+s2
        chart = deepcopy(chartx)
    if EndOfSent in chart[0]:
        return(1)
    else:
        return(0)


### use bigram prob to select the best sequnece
def chartparse_ngram():
    global chart
    global CProb

    CProb.clear()


    for j in range(1,EndOfSent):
        chartx = deepcopy(chart)
        if j in chart[0]:
            s1 = chart[0][j]
            for k in chart[j]:
                    s2 = chart[j][k]
#                    print 0,j,k
                    if k not in chart[0]:                        
                        chartx[0][k] = s1+s2
                        CProb[k] = BigramProb(s1+s2)
#                        print(s1+s2,'new',CProb[k])
                    else:
#                        print(s1+s2,BigramProb(s1+s2),CProb[k])
                        if BigramProb(s1+s2) > CProb[k]:
                            chartx[0][k] = s1+s2
                            CProb[k] = BigramProb(s1+s2)
#                            print(s1+s2,'old',CProb[k])
        chart = deepcopy(chartx)
    if EndOfSent in chart[0]:
        return(1)
    else:
        return(0)

def BigramProb(WLst):
    global CProb

    p=1.
    for i in range(len(WLst)-1):
        cx = tltk.corpus.bigram(WLst[i],WLst[i+1])
        if cx > 0.:
            p += math.log(cx/1000000)
        else:
            p += math.log(0.0001/1000000)    

    return(p)

##########################################
# Compute Collocation Strength between w1,w2
# stat = chi2 | mi | ll
##########################################
def compute_colloc(stat,w1,w2):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global NoTrigram
    global TotalWord
    global TotalLex

    if BiCount[(w1,w2)] < 1 or Count[w1] < 1 or Count[w2] < 1:
        BiCount[(w1,w2)] +=1
        Count[w1] +=1
        Count[w2] +=1 
        TotalWord +=2
#    print(w1,w2,Count[w1],Count[w2],BiCount[(w1,w2)],TotalWord)    
###########################
##  Mutual Information
###########################
    if stat == "mi":
        mi = float(BiCount[(w1,w2)] * TotalWord) / float((Count[w1] * Count[w2]))
        value = abs(math.log(mi,2))
#########################
### Compute Chisquare
##########################
    if stat == "chi2":
        value=0
        O11 = BiCount[(w1,w2)]
        O21 = Count[w2] - BiCount[(w1,w2)]
        O12 = Count[w1] - BiCount[(w1,w2)]
        O22 = TotalWord - Count[w1] - Count[w2] +  BiCount[(w1,w2)]
        value = float(TotalWord * (O11*O22 - O12 * O21)**2) / float((O11+O12)*(O11+O21)*(O12+O22)*(O21+O22))

    return(value)
    
##############################################################################    
########  create each unit (char/syllable) as a possible edge for chart parsing
def gen_unknown_w(SylLst):
    global chart
    global EndOfSent

    for i in range(EndOfSent):
        chart[i][i+1] = [SylLst[i]]
        if SylLst[i] not in TDICT:
            for j in range(i+1,EndOfSent):
                if SylLst[j] not in TDICT:
                    newwrd = ''.join(SylLst[i:j+1])
                    chart[i][j+1] = [newwrd]
#                    print('Create',SylLst[i],SylLst[j],newwrd)
                else:
                    break    
    return(1)


def gen_unknown_thaiw(SylLst):
    global chart
    global EndOfSent

    for i in range(EndOfSent):
        chart[i][i+1] = [SylLst[i]]
### add one unit that might be misspelled
        if SylLst[i] not in TDICT:
            for newwrd in spell_candidates(SylLst[i]):
                    if newwrd in TDICT:
#                        print(SylLst[i],'1=>',newwrd)
                        chart[i][i+1] = [newwrd]            
### add two or three consecutive units that might be misspelled
        if ''.join(SylLst[i:i+2]) not in TDICT:
           for newwrd in spell_candidates(''.join(SylLst[i:i+2])):
                    if newwrd in TDICT:
#                        print(SylLst[i:i+2],'2=>',newwrd)
                        chart[i][i+2] = [newwrd]
        if ''.join(SylLst[i:i+3]) not in TDICT:
           for newwrd in spell_candidates(''.join(SylLst[i:i+3])):
                    if newwrd in TDICT:
#                        print(SylLst[i:i+3],'3=>',newwrd)
                        chart[i][i+3] = [newwrd]
                        
    return(1)

####  gen a word from a sequence of English abbreviation
####  e.g.  เอบีเอ็น เอ็นบีเค  
def eng_abbr(SylLst):
    global chart
    global EndOfSent
    global EngAbbr
    i=0
    while i < EndOfSent-1:
        if SylLst[i] in EngAbbr:
            j=i+1
            while j<EndOfSent and SylLst[j] in EngAbbr:
                j=j+1
            if j>i+1:
                chart[i][j] = [''.join(SylLst[i:j])]
#                print(SylLst[i:j],'=>EngAbbr')
                i=j+1
            i=i+1    
        else:
            i=i+1
    return(1)



#############################################################################################################
#########  Chart Parsing, ceate a new edge from two connected edges, always start from 0 to connect {0-j} + {j+k}
#########  If maximal collocation appraoch is chosen, the sequence with highest score will be selected
def chart_parse():
    global chart
    global CollocSt
    
    for j in range(EndOfSent):
        chartx = deepcopy(chart)
        if j in chart[0]:
            s1 = chart[0][j]
            for k in chart[j]:
                    s2 = chart[j][k]
                    if k not in chart[0]:                        
                        chartx[0][k] = s1+s2
#                        CollocSt[(0,k)] = (CollocSt[(0,j)] + CollocSt[(j,k)])/2.0
                        CollocSt[(0,k)] = CollocSt[(0,j)] + CollocSt[(j,k)]
                    else:
                        if CollocSt[(0,j)]+CollocSt[(j,k)] > CollocSt[(0,k)]:
#                            CollocSt[(0,k)] = (CollocSt[(0,j)] + CollocSt[(j,k)])/2.0
                            CollocSt[(0,k)] = CollocSt[(0,j)] + CollocSt[(j,k)]
                            chartx[0][k] = s1+s2
        chart = deepcopy(chartx)
    if EndOfSent in chart[0]:
        return(1)
    else:
        return(0)


#############################################################################################################
###  Syllable Segmentation for Thai texts
### Input = a paragraph of Thai texts
def syl_segment(Input):
    global SegSep
    global SSegSep
    output = ""
    out = ""
    
    Input = preprocess(Input)
    sentLst = Input.split(SegSep)
    for s in sentLst:
#        print "s:",s
        inLst = s.split(SSegSep)
        for inp in inLst:
            if inp == '': continue            
            objMatch = re.match(r"[^ก-์]+",inp)
            if objMatch:
                out = inp
            else:
                out = sylseg(inp)
            output = output+out+SylSep
#        output = output.rstrip(SylSep)
        output = output+'<s/>'    ####write <s/> output for SegSep   
    return(output)        

#############################################################################################################
####### Segment syllable using trigram statistics, only strings matched with a defined syllable pattern will be created
####  Input = Thai string
def sylseg(Input):
    global SylSep
    global PRON
    
    schart = defaultdict(dict)
    probEnd = defaultdict(float)
    schartx = {}
    schart.clear()
    probEnd.clear()
    tmp = []
    
    EndOfInput = len(Input)
    for f in PRON:
        compiled_f = _RE_CACHE.get(f)
        if compiled_f is None:
            compiled_f = _RE_CACHE[f] = re.compile(f)
        for i in range(EndOfInput):
            Inx = Input[i:]
            matchObj = compiled_f.match(Inx)
            if matchObj:
                k=i+len(matchObj.group())
                schart[i][k] = [matchObj.group()]
                probEnd[(i,k)] = prob_trisyl([matchObj.group()])
#                print("match",i,k, matchObj.group(),f,probEnd[(i,k)])
    
    for j in range(EndOfInput):
        schartx = deepcopy(schart)
        if j in schart[0]:
            s1 = schart[0][j]
            for k in schart[j]:
                    s2 = schart[j][k]
                    ####****** change this to merge only form, need to do this, otherwise probtrisyl is not correct.
                    tmp = mergekaran(s1+s2)
                    if k not in schart[0]:                        
#                        schartx[0][k] = s1+s2
#                        probEnd[k] = prob_trisyl(s1+s2)
                        schartx[0][k] = tmp
                        probEnd[(0,k)] = prob_trisyl(tmp)
#                        print("new",tmp,probEnd[k])
                    else:
#                        p = prob_trisyl(s1+s2)
                        p = prob_trisyl(tmp)
                        if p > probEnd[(0,k)]:
#                            print("replace",tmp,p,probEnd[(0,k)])
#                            schartx[0][k] = s1+s2 
                            schartx[0][k] = tmp 
                            probEnd[(0,k)] = p
        schart = deepcopy(schartx)
    if EndOfInput in schart[0]:    
        return(SylSep.join(schart[0][EndOfInput]))
    else:
        return('<Fail>'+Input+'</Fail>')

######################
def mergekaran(Lst):
####  reconnect karan part to the previous syllable for SylSegment
   rs = []
   Found = 'n'
   Lst.reverse()
   for s in Lst:
        if re.search(r"(.+)[ิุ]์",s):    # anything + i or u + Karan
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        elif  re.search(r"(.+)์",s):  # anything + Karan
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        if Found == 'y':
            s += x
            rs.append(s)
            Found = 'n'
        else:
            rs.append(s)
   rs.reverse()
   return(rs)

def mergekaran1(Lst):
####  reconnect karan part to the previous syllable for SylSegment
#### include merhing pronunciation
   rs = []
   global MKaran
   MKaran.clear()
   Found = 'n'
   Lst.reverse()
   for s in Lst:
        if re.search(r"(.+)[ิุ]์",s):    # anything + i or u + Karan
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        elif  re.search(r"(.+)์",s):  # anything + Karan
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        if Found == 'y':
            for ph in PRONUN[s]:
                if (s+x,ph) not in MKaran:
                    PRONUN[s+x].append(ph)
                    MKaran[(s+x,ph)] = 1 
            s += x
            rs.append(s)
            Found = 'n'
        else:
            rs.append(s)
   rs.reverse()
   return(rs)

########################################
# calculate proability of each possible output
#  Version 1.6>  expect input = list of forms
########################################
def prob_trisyl(SylLst):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global NoTrigram
    global TotalWord
    global TotalLex
    global SegSep
    Prob = defaultdict(float)
    
#    SegSep = chr(127)

    pw2 = SegSep
    pw1 = SegSep
    Probx = 1.0
    
    for w in SylLst:
        if (w,pw1,pw2) in Prob:
            Proba = Prob[(w,pw1,pw2)]
        else:
            Prob[(w,pw1,pw2)] = prob_wb(w,pw1,pw2)
            Proba = Prob[(w,pw1,pw2)]
#        Probx *= Proba
        Probx += Proba    ## prob is changed to log
        pw2 = pw1
        pw1 = w
#    print("prob ",Probx)
    
    return(Probx)

########################################
# p(w | pw2 pw1)   Smoothing trigram prob  Witten-Bell
#######################################
def prob_wb(w,pw1,pw2):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global NoTrigram
    global TotalWord
    global TotalLex
    
    p3 = 0.0
    p2 = 0.0
    p1 = 0.0
    p = 0.0
    px1 = 0.0
    
#    print("trigram ", '|',pw2,'|',pw1,'|',w)
#    print("count ",TriCount[(pw2,pw1,w)],BiCount[(pw1,w)],Count[w])
    if TriCount[(pw2,pw1,w)] > 0:
        p3 = float(TriCount[(pw2,pw1,w)]) / float( BiCount[(pw2,pw1)] + BiType[(pw2,pw1)])
    if BiCount[(pw1,w)] > 0:
        p2 = float( BiCount[(pw1,w)]) / float((Count[pw1] + Type[pw1]) )
    if Count[w] > 0:
        p1 = float( Count[w]) / float(TotalWord + TotalLex)
    p = 0.8 * p3 + 0.15 * p2 + 0.04 * p1 + 1.0 / float((TotalWord + TotalLex)**2)
### change to log to prevent underflow value which can cause incorrect syllable segmentation
    p = math.log(p)
#    print('prob',p)

    return(p)

    

###### Read syllable pattern
def read_sylpattern(Filename):
    global PRON
    global stable
    global AK
    global MKaran
    global EngAbbr
    
    stable = defaultdict(defaultdict)
    AK = defaultdict(str)
    MKaran = defaultdict(int)
#    PRON = defaultdict(list)
#    PRON = defaultdict(str)
    
    tmp = [] 
    file1 = open(Filename,'r',encoding = 'cp874')
    for line in file1:
        if re.match(r'#',line):
            continue
        line = line.rstrip()
        tmp = line.split(',')
        tmp[0] = re.sub(r"X",u"([ก-ฮ])",tmp[0])
        tmp[0] = re.sub(r"C",u"([กขคจดตทบปผพฟสศซ])",tmp[0])
        tmp[0] = re.sub(r'Y',u"([ก-ฬฮ])",tmp[0])
        tmp[0] = re.sub(r'R',u"([รลว])",tmp[0])
        tmp[0] = re.sub(r'K',u"([ก-ฮ])",tmp[0])
        tmp[0] = re.sub(r'A',u"([กจฆดตบปอขฉฐถผฝศษสหคชพภทธมยรลวนณซญฑฏฌ])",tmp[0])
        tmp[0] = re.sub(r'Z',u"([กงดนมบรลฎฏจตณถพศทสชคภปญ])",tmp[0])
        tmp[0] = re.sub(r'D',u"([กงดนมบวยต])",tmp[0])
        tmp[0] = re.sub(r'W',u"[ก-ฮ]",tmp[0])
        tmp[0] = re.sub(r'\.',u"[\.]",tmp[0])
#        re.sub('Q',u"[\(\)\-\:\'\xCF\xE6]",tmp[0])
        if tmp[2] == "T":
            tmp[0] = re.sub(r"T",u"[่้๊๋]",tmp[0])
        else:
            tmp[0] = re.sub(r"T",u"[่้๊๋]*",tmp[0])
            
#       print tmp[0]
#        PRON[tmp[0]] = tmp[1]
        PRON[tmp[0]].append(tmp[1])
    
#    for f in PRON:
#        for x in PRON[f]:
#            print(f,x)
    stable['X'] = { 'ก' : 'k', 'ข' : 'kh' , 'ฃ':'kh', 'ค' : 'kh', 'ฅ' : 'kh','ฆ' : 'kh', 'ง' : 'N', 'จ' : 'c', 'ฉ' : 'ch', 'ช' : 'ch', 'ซ' : 's', 'ฌ' : 'ch','ญ' : 'j','ฎ' : 'd','ฏ' : 't','ฐ' : 'th','ฑ' : 'th','ฒ' : 'th','ณ' : 'n','ด' : 'd','ต' : 't','ถ' : 'th','ท' : 'th','ธ' : 'th','น' : 'n','บ' : 'b','ป' : 'p','ผ' : 'ph','ฝ' : 'f','พ' : 'ph','ฟ' : 'f','ภ' : 'ph','ม' : 'm','ย' : 'j','ร' : 'r','ฤ' : 'r','ล' : 'l','ฦ' : 'l','ว' : 'w','ศ' : 's','ษ' : 's','ส' : 's','ห' : 'h','ฬ' : 'l','อ' : '?','ฮ' : 'h' }
    stable['Y'] = { 'ก' : 'k', 'ข' : 'k' , 'ค' : 'k', 'ฆ' : 'k', 'ง' : 'N', 'จ' : 't', 'ฉ' : '-', 'ช' : 't', 'ซ' : 't', 'ฌ' : '-','ญ' : 'n','ฎ' : 't','ฏ' : 't','ฐ' : 't','ฑ' : 't','ฒ' : 't','ณ' : 'n','ด' : 't', 'ต' : 't','ถ' : 't','ท' : 't','ธ' : 't','น' : 'n','บ' : 'p','ป' : 'p','ผ' : '-','ฝ' : '-','พ' : 'p','ฟ' : 'p','ภ' : 'p','ม' : 'm','ย' : 'j','ร' : 'n','ฤ' : '-','ล' : 'n','ฦ' : '-','ว' : 'w','ศ' : 't','ษ' : 't','ส' : 't' ,'ห' : '-','ฬ' : 'n','อ' : '-','ฮ' : '-' }

    stable['A'] = stable['X']
    stable['K'] = stable['X']
    stable['C'] = stable['X']
    stable['R'] = stable['X']
    stable['G'] = stable['X']
    stable['E'] = stable['X']
    
    stable['D'] = stable['Y']
    stable['Z'] = stable['Y']
    stable['H'] = stable['Y']
    stable['F'] = stable['Y']

    AK['ก'] = "นฎฐณตถบปมรลวษสพหฬ"
    AK['ข'] = "จนณบมยรลฬดตทษภส"
    AK['ค'] = "กชดตนณปมทฑหบ"
    AK['ฆ'] = "กบรสช"
    AK['จ'] = "ณตนมรลทยกด"
    AK['ฉ'] = "กงนบพรลวทม"
    AK['ช'] = "กญนยรลวฎคทบมอ"
    AK['ซ'] = "ล"
    AK['ญ'] = "กภลญ"
    AK['ณ'] = "กร"
    AK['ด'] = "นรมวบ"
    AK['ต'] = "กตนภยรลฆงถบมวฤท"
    AK['ถ'] = "กนลวศพงมร"
    AK['ท'] = "กชนบมยรลวสหงศ"
    AK['ธ'] = "นภมยรวกพช"
    AK['น'] = "กทธฎภมยรวคขปลวห"
    AK['บ'] = "ดรทพม"
    AK['ป'] = "กฐฏณทนภรวศถฎตปยสหด"
    AK['ผ'] = "งจชดนลสวกคณทยรอ"
    AK['ฝ'] = "ร"
    AK['พ'] = "กญนมยลวสหณจธ"
    AK['ภ'] = "ยรคณมฤวน"
    AK['ม'] = "กณตนยรลหศดธมสฆว"
    AK['ย'] = "กดธภวสบศตมนถช"
    AK['ร'] = "กจณดพภมยวสหชตถนบ"
    AK['ล'] = "กคฆดตอบปพมลวห"
    AK['ว'] = "ชณดนพภรลสมยกจฏตทธปฤศหคธ"
    AK['ศ'] = "จณนบรลวศพดธกตมยส"
    AK['ษ'] = "ณฎบภรนคธม"
    AK['ส'] = "กคงดตถนบปพภมยรลวหอจฟสขทธฤ"
    AK['ห'] = "กงพทนรภญนมยรลวบต"
    AK['อ'] = "กงชณดตธนพภมยรลวสศคบฆจทปห"
    AK['ฑ'] = "มสรนค"
    AK['ฐ'] = "กจ"
    AK['ฏ'] = "ก"
    AK['ฌ'] = "ก"

    EngAbbr = ['เอ','บี','ซี','ดี','อี','เอฟ','จี','เจ','เอช','ไอ','เค','แอล','เอ็ม','เอ็น','โอ','พี','คิว','อาร์','เอส','ที','ยู','วี','เอ็กซ์','เอ็ก','วาย','แซด']
    ## cannot add 'ดับบลิว' because it has two syllables

    return(1)


##########  Read syllanle dict, pronunciation not conformed to sylrule is specified here
def read_syldict(Filename):
    global PRON
    
    file1 = open(Filename,'r',encoding='cp874')
    for line in file1:
        if re.match(r'#',line):
            continue
        line = line.rstrip()
        tmp = line.split("\t")
#        PRON[tmp[0]] = tmp[1]
        PRON[tmp[0]].append(tmp[1])
    return(1)

##########  Read syllable pattersn and pronunciation table
def read_PhSTrigram(File):
    global PhSTrigram
    global FrmSTrigram
    global PhSBigram
    global FrmSBigram
    global PhSUnigram
    global FrmSUnigram
    global AbsUnigram
    global AbsFrmSUnigram
    
    PhSTrigram = defaultdict(float)
    FrmSTrigram = defaultdict(float)
    PhSBigram = defaultdict(float)
    FrmSBigram = defaultdict(float)
    PhSUnigram = defaultdict(float)
    FrmSUnigram = defaultdict(float)
    AbsUnigram = defaultdict(float)
    AbsFrmSUnigram = defaultdict(float)
    
    IFile = open(File,'r',encoding='cp874')
    for line in IFile.readlines():
        line = line.rstrip()
        line = re.sub(r"<w>","|",line)
        (x, ct) = line.split('\t')
        (fs,p) = x.split('/')
        (x1,x2,x3) = fs.split('-')
#        print('read ',x1,x2,x3,p,ct)
        PhSTrigram[(x1,x2,x3,p)] += float(ct)
        FrmSTrigram[(x1,x2,x3)] += float(ct)
        PhSBigram[(x1,x2,p)] += float(ct)
        FrmSBigram[(x1,x2)] += float(ct)
        PhSBigram[(x2,x3,p)] += float(ct)
        FrmSBigram[(x2,x3)] += float(ct)
        PhSUnigram[(x2,p)] += float(ct)
        FrmSUnigram[x2] += float(ct)
        abs_x2 = re.sub(r"[่้๊๋]","",x2)
        abs_x2 = re.sub(r"[ก-ฮ]","C",abs_x2)
        abs_p = re.sub(r"[0-9]","",p)
        abs_p = re.sub(r"[^aeio@OuxU]","C",abs_p)
#        print(x2,'=>',abs_x2,':',p,'=>',abs_p)
        AbsUnigram[(abs_x2,abs_p)] += float(ct)
        AbsFrmSUnigram[abs_x2] += float(ct)
    IFile.close()

#### read syllable variants file
def read_sylvar(Filename):
    global SYLVAR

    fileObject = open(Filename,'rb')  
    SYLVAR = pickle.load(fileObject)
    fileObject.close()
    return(1)

####  read trigram statistics file
def read_stat(Filename):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global TotalWord
    global TotalLex
    global TotalWord
    global TotalLex

    TriCount = defaultdict(int)
    BiCount = defaultdict(int)
    BiType = defaultdict(int)
    Count = defaultdict(int)
    Type = defaultdict(int)
    
    TotalWord = 0
    TotalLex = 0
    TriCount.clear()
    BiCount.clear()
    Count.clear()
    BiType.clear()
    Type.clear()


    fileObject = open(Filename,'rb')  
    TriCount = pickle.load(fileObject)
    for (X,Y,Z) in TriCount:
        BiType[(X,Y)] += 1
        BiCount[(X,Y)] += TriCount[(X,Y,Z)]
        Count[Y] += TriCount[(X,Y,Z)]

    for (X,Y) in BiCount:
        Type[X] += 1
        
    for X in Count:
        TotalLex += 1
        TotalWord += Count[X]
        
    return(1)
    

########## Preprocess Thai texts  #### adding SegSep and <s> for speocal 
def preprocess(input):
    global SegSep
    global SSegSep

    input = re.sub(r" +ๆ",r"ๆ",input)
    input = re.sub(r"ๆ([^ ])",r"ๆ"+SegSep+r"\1",input)  ## add space after ๆ
#    print('adjust',input,'xxx')

#    input = re.sub(u"เเ",u"แ",input)
####### codes suggested by Arthit Suriyawongkul #####
    NORMALIZE_DICT = [
        ('\u0E40\u0E40', '\u0E41'), # Sara E + Sara E -> Sara AE
        ('\u0E4D\u0E32', '\u0E33'), # Nikhahit + Sara AA -> Sara AM
        ('\u0E24\u0E32', '\u0E24\u0E45'), # Ru + Sara AA -> Ru + Lakkhangyao
        ('\u0E26\u0E32', '\u0E26\u0E45'), # Lu + Sara AA -> Lu + Lakkhangyao
    ]
    for k, v in NORMALIZE_DICT:
        input = input.replace(k, v)
########################################################        
#    print input.encode('raw_unicode_escape')

  ###  handle Thai writing one character one space by deleting each space
#    pattern = re.compile(r'([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]{,2}) +|([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]{,2})$')
    pattern = re.compile(r'([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +|([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*)$')
#    while re.search(pattern, input):
#       input = re.sub(pattern, r"\1\2", input,count=1)
    input = re.sub(pattern, r"\1\2", input)

  ##### change space\tab between [ET][ET] and [ET]  to be SegSep
#    input = re.sub(r"([^\s\t\u00A0][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*[^\s\t\u00A0][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*)[\s\t\u00A0]+([^\s\t\u00A0])",r"\1"+SegSep+r"\2",input)
    input = re.sub(r"([^\s\t\u00A0]{3,})[\s\t\u00A0]+([^\s\t\u00A0]+?)",r"\1"+SegSep+r"\2",input)
#    print('1. ',input)
    
   ##### change space\tab between [ET] and [ET][ET]  to be SegSep
#    input = re.sub(r"([^\s\t\u00A0][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*)[\s\t\u00A0]+([^\s\t\u00A0][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*[^\s\t\u00A0][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*)",r"\1"+SegSep+r"\2",input)
    input = re.sub(r"([^\s\t\u00A0]+)[\s\t\u00A0]+([0-9]+)",r"\1"+SegSep+r"\2",input)
    input = re.sub(r"([^\s\t\u00A0]+?)[\s\t\u00A0]+([^\s\t\u00A0]{3,})",r"\1"+SegSep+r"\2",input)



        ### handle English and Thai mixed without a space inside $s by adding SSegSep (softSegSep)
    input = re.sub(r"([ก-์][ฯๆ])",r"\1"+SSegSep,input)
    input = re.sub(r"([\u0E01-\u0E5B]+\.?)([^\.\u0E01-\u0E5B\u001F]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"([^\.\u0E01-\u0E5B\u001F]+)([\u0E01-\u0E5B]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"(<.+?>)",SSegSep+r"\1",input)
    input = re.sub(r"([0-9a-zA-Z\.\-]{2,})([\u0E01-\u0E5B]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"(\.\.\.+)",r""+SSegSep+r"\1"+SSegSep,input)    #  ....  add SSegSep after that
#    print("3. ",input)

    return(input)



#############################################################################################################
### initialization by read syllable patterns, syllable trigrams, and satndard dictionary
def initial():
    global SylSep
    global WordSep
    global SegSep
    global SSegSep
    global TDICT
    global PRON
    global CProb
    global SYLVAR

    PRON = defaultdict(list)
    SYLVAR = defaultdict(list)
#    PRON = {}    
    TDICT = {}
    CProb = defaultdict(float)
    
    SylSep = chr(126)    ## ~
    WordSep = chr(124)   ## |
    SSegSep = chr(30)
    SegSep = chr(31)

    from ._data import data_path
    ATA_PATH = str(data_path('fallback'))

    read_sylpattern(ATA_PATH + '/sylrule.lts')
    read_syldict(ATA_PATH +  '/thaisyl.dict')
    read_stat(ATA_PATH + '/sylseg.3g')
    read_thdict(ATA_PATH +  '/thdict')
    read_sylvar(ATA_PATH + '/sylform_var.pkl')

    read_PhSTrigram(ATA_PATH +  '/PhSTrigram.sts')

    return(1)


############ END OF GENERAL MODULES 
initial()


##########################################################################
## testing area

#print(g2p('ราคาค่าตัววันนี้ไอซ์แลนด์'))
#print(word_segment('คอยสังเกตดูอาการของตัวเองด้วยนะซาศโฏะจัง'))
#print(th2roman('เด็กอยากไปโรงเรียน วันนี้ที่นี่เท่านั้น วันอื่นไม่ไป วันไหนก็ไม่ไป พรุ่งนี้ก็ไม่'))

