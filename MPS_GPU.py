from tensor_method_GPU import contra, svd_update, tensor_slide, tensor_add
import numpy as np
import torch # tested under pytorch 0.4.0
# import math
# # %matplotlib inline
# import matplotlib.pyplot as plt

# torch.manual_seed(1) # Fix seed of the random number generators
# np.random.seed(1)

class tensor_net:
    def __init__(self,image,learning_rate,links,max_bond,min_bond):
        self.image=image
        self.n=image.size(0)#number of spins
        self.m=image.size(1)#number of images
        self.learning_rate=learning_rate
        self.links=links
        self.order=range(self.n)
        self.order_left=[]
        self.order_right=[]
        self.tensors=[]
        self.going_righ=1
        self.current_site=0
        self.max_bond=max_bond
        self.min_bond=min_bond
        self.contraction_z=range(self.n)
        self.contraction_psi=range(self.n)
        self.order[0]=([0,1])
        self.order_left.append([0,8*self.n+1])
        self.merged_tensor=torch.zeros([2,2,2]).cuda()
        self.merged_idx=[]
        self.psi=[]
        self.Z=0
        self.nll_history=[-np.inf]

        for i in range(1,self.n):
            self.order[i]=([i*2-1,i*2,i*2+1])
            self.order_left.append([8*self.n+i*2-1,i*2,8*self.n+i*2+1])
        self.order_left[self.n-1]=([10*self.n-3,self.n*2-2])
        self.order[self.n - 1] = ([2 * self.n - 3, self.n * 2 - 2])

        for j in range(len(links)):
            self.order[links[j][0]]=self.order[links[j][0]]+[6*self.n+2*j+1]
            self.order[links[j][1]] = self.order[links[j][1]] + [6 * self.n + 2*j + 1]
            self.order_left[links[j][0]] = self.order_left[links[j][0]] + [14 * self.n + 2*j + 1]
            self.order_left[links[j][1]] = self.order_left[links[j][1]] + [14 * self.n + 2*j + 1]


    def orthogonalize(self,err,max_bond):
            u,v=svd_update(self.tensors[self.current_site],self.order[self.current_site],self.tensors[self.current_site+1],self.order[self.current_site+1],
                            0, self.going_righ,err,max_bond)
            self.tensors[self.current_site]=u
            self.tensors[self.current_site+1]=v

    def init_tensors(self):
        self.tensors.append(torch.rand([2,self.min_bond]).cuda())
        for i in range(1,self.n-1):
            self.tensors.append(torch.rand(self.min_bond,2,self.min_bond).cuda())
        self.tensors.append(torch.rand([self.min_bond,2]).cuda())
        for j in range(len(self.links)):
            sl=self.tensors[self.links[j][0]].size()
            sr=self.tensors[self.links[j][1]].size()
            self.tensors[self.links[j][0]]=torch.rand(list(sl)+list([2])).cuda()
            self.tensors[self.links[j][1]] = torch.rand(list(sr) + list([2])).cuda()
        self.going_righ=1
        for j in range(self.n-1):
            self.current_site=j
            self.orthogonalize(0.0001,np.inf)
        # self.going_righ=0
        # for j in np.arange(self.n-2,0,-1):
        #     self.current_site=j
        #     self.orthogonalize(0,np.inf)

    def contraction_update_all_left(self):
        v1=torch.Tensor([0,1]).cuda()
        v2=torch.Tensor([1,0]).cuda()

        self.going_righ=0
        merg,idm = contra(self.tensors[0], self.order_left[0], self.tensors[0],self.order[0])
        self.contraction_z[0]=([merg,idm])
        merg_psi=[]
        for j in range(self.m):
            if self.image[0,j]==1:#n*m
                mergv,idv=contra(self.tensors[0],self.order_left[0],v1,[0])
            else:
                mergv,idv=contra(self.tensors[0],self.order_left[0],v2,[0])
            merg_psi.append(mergv)
        self.contraction_psi[0]=([merg_psi,idv])

        for i in range(1,self.n-2):
            merg,idm=contra(merg, idm, self.tensors[i],self.order_left[i])
            merg, idm = contra(merg, idm, self.tensors[i], self.order[i])
            self.contraction_z[i]=([merg,idm])

            merg_psi = []
            for j in range(0,self.m):
                if self.image[i, j] == 1:  # n*m
                    mergv,idvm=contra(self.contraction_psi[i-1][0][j],idv,self.tensors[i], self.order_left[i])
                    mergv,idvm = contra(mergv,idvm, v1, [2*i])
                else:
                    mergv, idvm = contra(self.contraction_psi[i-1][0][j], idv, self.tensors[i], self.order_left[i])
                    mergv, idvm = contra(mergv, idvm, v2, [2 * i])
                merg_psi.append(mergv)
            idv=idvm
            self.contraction_psi[i]=([merg_psi,idvm])

    def contraction_updat_twosite(self):
        v1 = torch.Tensor([0, 1]).cuda()
        v2 = torch.Tensor([1, 0]).cuda()
        if self.going_righ==0:
            if self.current_site==self.n-2:
                merg,idm=contra(self.tensors[self.n-1],self.order[self.n-1],self.tensors[self.n-1],self.order_left[self.n-1])
                # self.contraction_z[self.current_site+1]=merg
                merg_psi=[]
                for j in range(self.m):
                    if self.image[self.n-1, j] == 1:  # n*m
                        mergv, idvm = contra(self.tensors[self.n-1], self.order_left[self.n-1], v1, [self.n*2-2])
                    else:
                        mergv, idvm = contra(self.tensors[self.n-1], self.order_left[self.n-1], v2, [self.n*2-2])
                    merg_psi.append(mergv)

            else:


                merg, idm = contra(self.tensors[self.current_site+1], self.order_left[self.current_site+1],
                                   self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
                merg, idm = contra(merg, idm, self.tensors[self.current_site+1], self.order[self.current_site+1])

                merg_psi = []
                for j in range(0, self.m):
                    if self.image[self.current_site+1, j] == 1:  # n*m
                        mergv, idvm = contra(self.tensors[self.current_site+1], self.order_left[self.current_site+1],
                                             self.contraction_psi[self.current_site + 2][0][j],self.contraction_psi[self.current_site + 2][1] )
                        mergv, idvm = contra(mergv, idvm, v1, [2 * self.current_site+2])
                    else:
                        mergv, idvm = contra(self.tensors[self.current_site + 1], self.order_left[self.current_site + 1],
                                             self.contraction_psi[self.current_site + 2][0][j],self.contraction_psi[self.current_site + 2][1])
                        mergv, idvm = contra(mergv, idvm, v2, [2 * self.current_site + 2])
                    merg_psi.append(mergv)

            self.contraction_psi[self.current_site+1]=([merg_psi,idvm])
            self.contraction_z[self.current_site+1]=([merg,idm])

        if self.going_righ == 1:
                if self.current_site == 0:
                    merg, idm = contra(self.tensors[0], self.order[0], self.tensors[0],self.order_left[0])
                    # self.contraction_z[self.current_site+1]=merg
                    merg_psi=[]
                    for j in range(self.m):
                        if self.image[0, j] == 1:  # n*m
                            mergv, idvm = contra(self.tensors[0], self.order_left[0], v1, [0])
                        else:
                            mergv, idvm = contra(self.tensors[0], self.order_left[0], v2, [0])
                        merg_psi.append(mergv)

                else:

                    merg, idm = contra(self.tensors[self.current_site ], self.order_left[self.current_site ],
                                       self.contraction_z[self.current_site -1][0],self.contraction_z[self.current_site-1][1])
                    merg, idm = contra(merg, idm, self.tensors[self.current_site ],
                                       self.order[self.current_site ])

                    merg_psi = []
                    for j in range(0, self.m):
                        if self.image[self.current_site , j] == 1:  # n*m
                            mergv, idvm = contra(self.tensors[self.current_site ], self.order_left[self.current_site ],
                                                 self.contraction_psi[self.current_site -1][0][j],self.contraction_psi[self.current_site -1][1])
                            mergv, idvm = contra(mergv, idvm, v1, [2 * self.current_site ])
                        else:
                            mergv, idvm = contra(self.tensors[self.current_site], self.order_left[self.current_site ],
                                                 self.contraction_psi[self.current_site -1][0][j],self.contraction_psi[self.current_site -1][1])
                            mergv, idvm = contra(mergv, idvm, v2, [2 * self.current_site ])
                        merg_psi.append(mergv)

                    self.contraction_psi[self.current_site ] = ([merg_psi,idvm])
                    self.contraction_z[self.current_site ] = ([merg,idm])

    def compute_Z(self):
        merg,idm=contra(self.tensors[self.current_site],self.order[self.current_site],
                    self.tensors[self.current_site+1],self.order[self.current_site+1])
        merg_l, idm_l = contra(self.tensors[self.current_site], self.order_left[self.current_site],
                           self.tensors[self.current_site + 1], self.order_left[self.current_site + 1])
        merg_m,idm_m=contra(merg,idm,merg_l,idm_l)
        if self.current_site==0:
            Z,idz=contra(merg_m,idm_m,self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
            # Z, idz = contra(Z, idz, merg_l,idm_l)
        else:
            if self.current_site==self.n-2:
                Z, idz = contra(merg_m, idm_m, self.contraction_z[self.current_site -1][0],
                                self.contraction_z[self.current_site -1][1])
                # Z, idz = contra(Z, idz, merg_l, idm_l)
            else:
                Z_l,idz_l=contra(merg_m, idm_m, self.contraction_z[self.current_site -1][0],
                                self.contraction_z[self.current_site -1][1])
                Z,idz=contra(Z_l,idz_l,self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
                # Z, idz = contra(Z, idz, merg_l, idm_l)
        return Z

    def compute_psi(self):
        Psi=[]
        v1=torch.Tensor([0,1]).cuda()
        v2=torch.Tensor([1,0]).cuda()
        if self.current_site==0:
            for j in range(self.m):
                if self.image[0, j] == 1:  # n*m
                    mergv, idvm = contra(self.tensors[0], self.order_left[0], v1, [0])
                    mergv, idvm = contra(mergv, idvm, self.tensors[1],self.order_left[1])
                    if self.image[1,j] == 1:
                        mergv, idvm = contra(mergv, idvm,v1, [2])
                    else:
                        mergv, idvm = contra(mergv, idvm, v2, [2])
                else:
                    mergv, idvm = contra(self.tensors[0], self.order_left[0], v2, [0])
                    mergv, idvm = contra(mergv, idvm, self.tensors[1], self.order_left[1])
                    if self.image[1, j] == 1:
                        mergv, idvm = contra(mergv, idvm, v1, [2])
                    else:
                        mergv, idvm = contra(mergv, idvm, v2, [2])


                psi,id=contra(mergv,idvm,self.contraction_psi[2][0][j],self.contraction_psi[2][1])
                Psi.append(psi)
        else:
            if self.current_site==self.n-2:
                for j in range(self.m):
                    if self.image[self.n-2, j] == 1:  # n*m
                        mergv, idvm = contra(self.tensors[self.n-2], self.order_left[self.n-2], v1, [2*self.n-4])
                        mergv, idvm = contra(mergv, idvm, self.tensors[self.n-1], self.order_left[self.n-1])
                        if self.image[self.n-1, j] == 1:
                            mergv, idvm = contra(mergv, idvm, v1, [2*self.n-2])
                        else:
                            mergv, idvm = contra(mergv, idvm, v2, [2*self.n-2])
                    else:
                        mergv, idvm = contra(self.tensors[self.n-2], self.order_left[self.n-2], v2, [2*self.n-4])
                        mergv, idvm = contra(mergv, idvm, self.tensors[self.n-1], self.order_left[self.n-1])
                        if self.image[self.n-1, j] == 1:
                            mergv, idvm = contra(mergv, idvm, v1, [2*self.n-2])
                        else:
                            mergv, idvm = contra(mergv, idvm, v2, [2*self.n-2])

                    psi, id = contra(mergv, idvm, self.contraction_psi[self.n-3][0][j],self.contraction_psi[self.n-3][1])
                    Psi.append(psi)

            else:
                for j in range(self.m):
                    if self.image[self.current_site, j] == 1:  # n*m
                        mergv, idvm = contra(self.tensors[self.current_site], self.order_left[self.current_site], v1, [2*self.current_site])
                        mergv, idvm = contra(mergv, idvm, self.tensors[self.current_site+1], self.order_left[self.current_site+1])
                        if self.image[self.current_site+1, j] == 1:
                            mergv, idvm = contra(mergv, idvm, v1, [2*self.current_site+2])
                        else:
                            mergv, idvm = contra(mergv, idvm, v2, [2*self.current_site+2])
                    else:
                        mergv, idvm = contra(self.tensors[self.current_site], self.order_left[self.current_site], v2,
                                             [2 * self.current_site])
                        mergv, idvm = contra(mergv, idvm, self.tensors[self.current_site + 1],
                                             self.order_left[self.current_site + 1])
                        if self.image[self.current_site + 1, j] == 1:
                            mergv, idvm = contra(mergv, idvm, v1, [2 * self.current_site + 2])
                        else:
                            mergv, idvm = contra(mergv, idvm, v2, [2 * self.current_site + 2])

                    mergv, idvm = contra(mergv, idvm, self.contraction_psi[self.current_site-1][0][j], self.contraction_psi[self.current_site-1][1])
                    psi,id=contra(mergv, idvm, self.contraction_psi[self.current_site+2][0][j], self.contraction_psi[self.current_site+2][1])
                    Psi.append(psi)

        return Psi

    def gradient_descent(self):
        dpsi=[]
        dPsi=[]
        merged_idx2 = self.merged_idx[:]
        for k in range(len(self.merged_idx)):
            if self.merged_idx[k] % 2 == 1:
                merged_idx2[k] = merged_idx2[k] - 8 * self.n

        if self.current_site==0:


            dZ,dZ_idx=contra(self.merged_tensor,merged_idx2,self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
            for j in range(self.m):
                dpsi.append(self.contraction_psi[self.current_site+2][0][j])
            dpsi_idx = self.contraction_psi[self.current_site + 2][1]
            dPsi.append(dpsi+[dpsi_idx])
        else:
            if self.current_site==self.n-2:
                dZ, dZ_idx = contra(self.merged_tensor, merged_idx2, self.contraction_z[self.current_site - 1][0],
                                    self.contraction_z[self.current_site -1][1])
                for j in range(self.m):
                    dpsi.append(self.contraction_psi[self.current_site-1][0][j])
                dpsi_idx = self.contraction_psi[self.current_site -1][1]
                dPsi.append(dpsi+[dpsi_idx])
            else:
                merg_Z,idz=contra(self.merged_tensor, merged_idx2, self.contraction_z[self.current_site - 1][0],
                                    self.contraction_z[self.current_site -1][1])
                dZ, dZ_idx = contra(merg_Z,idz, self.contraction_z[self.current_site + 2][0],
                                    self.contraction_z[self.current_site + 2][1])
                for j in range(self.m):
                    merg_psi,dpsi_idx=contra(self.contraction_psi[self.current_site-1][0][j],self.contraction_psi[self.current_site -1][1],
                    self.contraction_psi[self.current_site + 2][0][j], self.contraction_psi[self.current_site +2][1])
                    dpsi.append(merg_psi)
                dPsi.append(dpsi+[dpsi_idx])
        dmerge=torch.zeros((self.merged_tensor).size()).cuda()

        for im1 in [1,2]:
            for im2 in [1,2]:
                dpsi=torch.zeros((dPsi[0][0]).size()).cuda()
                im=(self.image[self.current_site,:]==im1)*(self.image[self.current_site+1,:]==im2)
                for j in range(self.m):
                    if im[j]==1:
                        dpsi=2*dPsi[0][j]/self.psi[j]+dpsi
                    dmerge=tensor_slide(dmerge,self.merged_idx,[2-im1,2-im2],[self.current_site*2,self.current_site*2+2],dpsi,(dPsi[0][-1]))
        dmerge=tensor_add(dmerge/self.n,self.merged_idx,-2*dZ/self.Z,dZ_idx)

        gnorm = torch.norm(dmerge) / 20
        if (gnorm < 1.0): #% & & self.bond_dims(self.current_bond) <= 50;
            dmerge = dmerge / gnorm;
        dmerge = self.merged_tensor + self.learning_rate * dmerge


        return dmerge

    def gradient_descent5(self,j,k):
        self.current_site=j
        self.merged_tensor,self.merged_idx=contra(self.tensors[j],self.order_left[j],self.tensors[k],self.order_left[k])
        merged_idx2 = self.merged_idx[:]
        for l in range(len(self.merged_idx)):
            if self.merged_idx[l] % 2 == 1:
                merged_idx2[l] = merged_idx2[l] - 8 * self.n

        mer_l,mer_l_idx=contra(self.tensors[0],self.order_left[0],self.tensors[0],self.order[0])
        for i in range(1,j):
            mer_l,mer_l_idx=contra(mer_l,mer_l_idx,self.tensors[i],self.order_left[i])
            mer_l, mer_l_idx = contra(mer_l, mer_l_idx, self.tensors[i], self.order[i])

        mer_r, mer_r_idx = contra(self.tensors[self.n-1], self.order_left[self.n-1], self.tensors[self.n-1], self.order[self.n-1])
        for i in np.arange(self.n-2,k,-1):
            mer_r, mer_r_idx = contra(mer_r, mer_r_idx, self.tensors[i], self.order_left[i])
            mer_r, mer_r_idx = contra(mer_r, mer_r_idx, self.tensors[i], self.order[i])

        mer_m,mer_m_idx=contra(self.tensors[j+1],self.order_left[j+1],self.tensors[j+1],self.order[j+1])
        for i in range(j+2,k):
            mer_m, mer_m_idx = contra(mer_m, mer_m_idx, self.tensors[i], self.order_left[i])
            mer_m, mer_m_idx = contra(mer_m, mer_m_idx, self.tensors[i], self.order[i])

        dZ,dZ_idx=contra(mer_l,mer_l_idx,self.merged_tensor,merged_idx2)
        dZ,dZ_idx=contra(dZ,dZ_idx,mer_r,mer_r_idx)
        dZ,dZ_idx=contra(dZ,dZ_idx,mer_m,mer_m_idx)

        dpsi=[]
        dPsi=[]
        v1=torch.Tensor([0,1]).cuda()
        v2=torch.Tensor([1,0]).cuda()
        for jl in range(self.m):
            if self.image[0,jl]==1:
                psi_l,psi_l_idx =contra(self.tensors[0],self.order_left[0],v1,[0])
            else:
                psi_l, psi_l_idx = contra(self.tensors[0], self.order_left[0], v2, [0])
            dpsi.append(psi_l)
        for i in range(1,j):
            for jl in range(self.m):
                dpsi[jl], psi_l_idx2 = contra(dpsi[jl], psi_l_idx, self.tensors[i],self.order_left[i])
                if self.image[j, jl] == 1:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v1, [2*i])
                else:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v2, [2*i])
                dpsi[jl]=psi_l
            psi_l_idx=psi_l_idx2[:]
        for i in range(j+1,k):
            for jl in range(self.m):
                dpsi[jl], psi_l_idx2 = contra(dpsi[jl], psi_l_idx, self.tensors[i], self.order_left[i])
                if self.image[j, jl] == 1:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v1, [2*i])
                else:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v2, [2*i])
                dpsi[jl]=psi_l
            psi_l_idx = psi_l_idx2[:]
        for i in range(k+1,self.n):
            for jl in range(self.m):
                dpsi[jl], psi_l_idx2 = contra(dpsi[jl], psi_l_idx, self.tensors[i], self.order_left[i])
                if self.image[j, jl] == 1:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v1, [2*i])
                else:
                    psi_l, psi_l_idx2  = contra(dpsi[jl], psi_l_idx2 , v2, [2*i])
                dpsi[jl]=psi_l
            psi_l_idx = psi_l_idx2[:]
        dPsi.append(dpsi + [psi_l_idx ])
        dmerge = torch.zeros((self.merged_tensor).size()).cuda()

        for im1 in [1, 2]:
            for im2 in [1, 2]:
                dpsi = torch.zeros((dPsi[0][0]).size()).cuda()
                im = (self.image[self.current_site, :] == im1) * (self.image[self.current_site + 1, :] == im2)
                for jl in range(self.m):
                    if im[jl] == 1:
                        dpsi = 2 * dPsi[0][jl] / self.psi[jl] + dpsi
                    dmerge = tensor_slide(dmerge, self.merged_idx, [2 - im1, 2 - im2],
                                          [j * 2, k * 2], dpsi, (dPsi[0][-1]))
        dmerge = tensor_add(dmerge / self.n, self.merged_idx, -2 * dZ / self.Z, dZ_idx)

        gnorm = torch.norm(dmerge) / 20
        if (gnorm < 1.0):  # % & & self.bond_dims(self.current_bond) <= 50;
            dmerge = dmerge / gnorm;
        dmerge = self.merged_tensor + self.learning_rate * dmerge

        return dmerge

    def train(self,loops):
        for loop in range(loops):
            self.going_righ=0
            for i in np.arange(self.n - 2, 0, -1):

                self.current_site=i
                self.merged_tensor,self.merged_idx=contra(self.tensors[self.current_site],self.order_left[self.current_site],
                                      self.tensors[self.current_site+1],self.order_left[self.current_site+1])
                self.Z=self.compute_Z()
                self.psi=self.compute_psi()
                dmerge=self.gradient_descent()
                # nll=0
                # for k in range(self.m):
                #     nll=nll+(torch.log(self.psi[k] *self.psi[k] /self.Z))
                # nll=nll/self.m
                # print nll,i
                self.tensors[self.current_site], self.tensors[self.current_site + 1] = svd_update(
                    self.tensors[self.current_site], self.order[self.current_site],
                    self.tensors[self.current_site + 1], self.order[self.current_site + 1], dmerge,
                    self.going_righ, 0.0002, self.max_bond
                )
                self.contraction_updat_twosite()

            self.going_righ = 1
            for i in range(self.n - 2):
                self.current_site = i
                self.merged_tensor, self.merged_idx = contra(self.tensors[self.current_site],
                                                             self.order_left[self.current_site],
                                                             self.tensors[self.current_site + 1],
                                                             self.order_left[self.current_site + 1])
                self.Z = self.compute_Z()
                self.psi = self.compute_psi()
                # nll = 0
                # for k in range(self.m):
                #     nll = nll + ( torch.log(self.psi[k] *self.psi[k]  / self.Z))
                # nll=nll/self.m
                # print nll,i
                dmerge = self.gradient_descent()
                self.tensors[self.current_site], self.tensors[self.current_site + 1] = svd_update(
                    self.tensors[self.current_site], self.order[self.current_site],
                    self.tensors[self.current_site + 1], self.order[self.current_site + 1], dmerge,
                    self.going_righ, 0.0002, self.max_bond
                )
                self.contraction_updat_twosite()
            for j in range(len(self.links)):
                self.Z = self.compute_Z()
                self.psi = self.compute_psi()
                k0 = self.links[j][0]
                k1=self.links[j][1]
                dmerge=self.gradient_descent5(k0,k1)
                self.tensors[k0], self.tensors[k1] = svd_update(
                    self.tensors[k0], self.order[k0],
                    self.tensors[k1], self.order[k1], dmerge,
                    self.going_righ, 0.0002, self.max_bond
                )
                self.contraction_update_all_left()
            nll=0
            for k in range(self.m):
                nll = nll + (torch.log(self.psi[k] *self.psi[k] / self.Z))
            nll = nll / self.m
            print nll
            if nll>self.nll_history[-1]+0.002:
                self.nll_history.append(nll)
            else:
                break




#
T=torch.rand([30,30])
T[T>0.5]=2
T[T<=0.5]=1
net=tensor_net(T,0.0001,np.array([[4,9]]),20,2)
net.init_tensors()
net.contraction_update_all_left()
net.train(20)
print net.nll_history