#####################################################################################################################
#  Date created : 10/21/2022
#  Date modified: 10/21/2022
#  Purpose      : To generate (full) node feature set from temporary features
#####################################################################################################################

import optparse, os, sys
import numpy as np
parser=optparse.OptionParser()
parser.add_option('-t', dest='t',
        default= '/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/input.txt',    #default empty!
        help= 'target list')
(options,args) = parser.parse_args()
target = options.t
def sigmoid(x):
    return 1/(1 + np.exp(-x))

f = open(target, 'r')
flines = f.readlines()
for line in flines:
                tgt = line.split('.')[0]
                print(tgt)
                name = tgt
                #tgtdata = []

                pdbfeat = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' + name +'.feat.npy')
                pdbfeat2 = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' + name + '.feat22.npy')
                pssmfeat = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/pssm/' + tgt + '.npy')
                ccountfeat = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' + name +'.concount.npy')
                pdbsincosangle = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' + name +'.feat_angle6.npy')
                pdbforwrevca = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' +  name +'.feat_forw_rev_ca6.npy')
                pdbimputed = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/temp/' + name +'.imputed3.npy')
                transfeat = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/input/' + name + '_trans.npy')
                esm2feat = np.load('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/esm/' + name + '_esm.npy')

                # feat33 = sigmoid(esm2feat[1:-1])
                # singlefeat = np.load('input/'+ name + 'msa_first_row.npy')

                maxlen = max(len(pdbfeat), len(pssmfeat))
                # singlefeat = sigmoid(singlefeat[:maxlen])
                #tgtdata = [[0 for _ in range(85)] for _ in range(maxlen)]
                tgtdata = np.zeros((maxlen, 3620))
                for ii in range(min(len(pdbfeat), len(pssmfeat))):
                        # # res = np.concatenate((pdbfeat[ii][:26], [pdbfeat[ii][27]], [ccountfeat[ii]], pdbfeat2[ii], pdbsincosangle[ii], pdbforwrevca[ii], pdbimputed[ii], pssmfeat[ii], feat33[ii], singlefeat[ii]))
                        # res = np.concatenate((pdbfeat[ii][:26], [pdbfeat[ii][27]], [ccountfeat[ii]], pdbfeat2[ii], pdbsincosangle[ii], pdbforwrevca[ii], pdbimputed[ii], pssmfeat[ii], feat33[ii]))
                        # #res_label = np.concatenate((res, label[ii]))
                        # tgtdata[ii] = res
                         if ii < len(pdbfeat) and ii < len(ccountfeat) and ii < len(pdbfeat2) and ii < len(pdbsincosangle) and ii < len(pdbforwrevca) and ii < len(pdbimputed) and ii < len(pssmfeat):
                                # print(f"ii = {ii}, pdbfeat size = {len(pdbfeat)}, ccountfeat size = {len(ccountfeat)}")
                                # res = np.concatenate((pdbfeat[ii][:26], [pdbfeat[ii][27]], [ccountfeat[ii]], pdbfeat2[ii], pdbsincosangle[ii], pdbforwrevca[ii], pdbimputed[ii], pssmfeat[ii], transfeat[ii], esm2feat[ii],feat33[ii]))
                                # res = np.concatenate((pdbfeat[ii][:26], [pdbfeat[ii][27]], [ccountfeat[ii]], pdbfeat2[ii], pdbsincosangle[ii], pdbforwrevca[ii], pdbimputed[ii], pssmfeat[ii], transfeat[ii], esm2feat[ii]))
                                res = np.concatenate(([ccountfeat[ii]], pdbsincosangle[ii], pdbforwrevca[ii], pdbimputed[ii], pssmfeat[ii], transfeat[ii], esm2feat[ii]))
                                tgtdata[ii] = res
                                
       
                
                # np.save('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing2/processed_features3/' + tgt + '.3669new', esm2feat)
                np.save('/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing2/processed_features_nopdb/' + tgt + '.3669new', tgtdata)

f.close()

