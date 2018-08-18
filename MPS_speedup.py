import numpy as np
import torch
from tensor_method import contra, svd_update, tensor_slide, tensor_add
import time
import scipy.io as sio

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
        self.image_tensors=[]
        self.going_righ=1
        self.current_site=0
        self.max_bond=max_bond
        self.min_bond=min_bond
        self.contraction_z=range(self.n)
        self.contraction_psi=range(self.n)
        self.contraction_psi2 = range(self.n)
        self.order[0]=([0,1])
        self.order_left.append([0,8*self.n+1])
        self.order_right.append([0,1])
        self.merged_tensor=torch.zeros([2,2,2],dtype=torch.float64)#.cuda()
        self.merged_idx=[]
        self.psi=[]
        self.Z=0
        self.nll_history=[-np.inf]

        for i in range(1,self.n):
            self.order[i]=([i*2-1,i*2,i*2+1])
            self.order_left.append([8*self.n+i*2-1,i*2,8*self.n+i*2+1])
            self.order_right.append([i*2-1,i*2,i*2+1])
        self.order_left[self.n-1]=([10*self.n-3,self.n*2-2])
        self.order[self.n - 1] = ([2 * self.n - 3, self.n * 2 - 2])
        self.order_right[self.n-1]=([2 * self.n - 3, self.n * 2 - 2,-1])
        # self.order_right[self.n]=[-1,-2]

        for j in range(len(links)):
            self.order[links[j][0]]=self.order[links[j][0]]+[6*self.n+2*j+1]
            self.order[links[j][1]] = self.order[links[j][1]] + [6 * self.n + 2*j + 1]
            self.order_left[links[j][0]] = self.order_left[links[j][0]] + [14 * self.n + 2*j + 1]
            self.order_left[links[j][1]] = self.order_left[links[j][1]] + [14 * self.n + 2*j + 1]

    def init_image_tensor(self):
        self.image_tensors.append(torch.zeros([2,self.m],dtype=torch.float64))
        for j in range(self.m):
            if self.image[0][j] == 1:
                self.image_tensors[0][1, j] = 1
            else:
                self.image_tensors[0][0, j] = 1
        for i in range(1,self.n):
            self.image_tensors.append(torch.zeros([self.m,2,self.m],dtype=torch.float64))
            for j in range(self.m):
                if self.image[i][j]==1:
                    self.image_tensors[i][j,1,j]=1
                else:
                    self.image_tensors[i][j,0,j]=1
        # self.image_tensors.append(torch.eye(self.m))
    def init_tensors(self):
        self.tensors.append(torch.ones([2,self.min_bond],dtype=torch.float64))
        for i in range(1,self.n-1):
            self.tensors.append(torch.ones(self.min_bond,2,self.min_bond,dtype=torch.float64))
        self.tensors.append(torch.ones([self.min_bond,2],dtype=torch.float64))
        for j in range(len(self.links)):
            sl=self.tensors[self.links[j][0]].size()
            sr=self.tensors[self.links[j][1]].size()
            self.tensors[self.links[j][0]]=torch.ones(list(sl)+list([2]),dtype=torch.float64)
            self.tensors[self.links[j][1]] = torch.ones(list(sr) + list([2]),dtype=torch.float64)
        self.going_righ=1
        for j in range(self.n-1):
            self.current_site=j
            self.orthogonalize(0.000,np.inf)
        # self.going_righ=0
        # for j in np.arange(self.n-2,0,-1):
        #     self.current_site=j
        #     self.orthogonalize(0,np.inf)
    def orthogonalize(self,err,max_bond):
            u,v=svd_update(self.tensors[self.current_site],self.order[self.current_site],self.tensors[self.current_site+1],self.order[self.current_site+1],
                            0, self.going_righ,err,max_bond)
            self.tensors[self.current_site]=u
            self.tensors[self.current_site+1]=v
    def contraction_update_all_left2(self):
        # v1=torch.Tensor([0,1])
        # v2=torch.Tensor([1,0])

        self.going_righ=0
        merg,idm = contra(self.tensors[0], self.order_left[0], self.tensors[0],self.order[0])
        self.contraction_z[0]=([merg,idm])
        merg_psi , merg_psi_idx= contra(self.tensors[0],self.order_left[0],self.image_tensors[0],self.order_right[0])
        self.contraction_psi2[0]=([merg_psi,merg_psi_idx])

        for i in range(1,self.n-2):
            merg,idm=contra(merg, idm, self.tensors[i],self.order_left[i])
            merg, idm = contra(merg, idm, self.tensors[i], self.order[i])
            self.contraction_z[i]=([merg,idm])

            merg_psi,merg_psi_idx=contra(merg_psi,merg_psi_idx,self.tensors[i],self.order_left[i])
            merg_psi, merg_psi_idx = contra(merg_psi, merg_psi_idx, self.image_tensors[i], self.order_right[i])
            self.contraction_psi2[i] = ([merg_psi, merg_psi_idx])

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
    def compute_psi2(self):
        merg_psi,idx=contra(self.tensors[self.current_site],self.order_left[self.current_site],
                    self.tensors[self.current_site+1],self.order_left[self.current_site+1])
        merg_psi_r,idx_r=contra(self.image_tensors[self.current_site],self.order_right[self.current_site],
                                self.image_tensors[self.current_site+1],self.order_right[self.current_site+1])
        merg,idx=contra(merg_psi,idx,merg_psi_r,idx_r)

        if self.current_site==0:
            psi,idx_psi=contra(merg,idx,self.contraction_psi2[2][0],self.contraction_psi2[2][1])
        else:
            if self.current_site == self.n - 2:
                psi, idx_psi = contra(merg, idx, self.contraction_psi2[self.current_site - 1][0],
                                self.contraction_psi2[self.current_site - 1][1])
                # Z, idz = contra(Z, idz, merg_l, idm_l)
            else:
                psi_r, idx_r = contra(merg, idx, self.contraction_psi2[self.current_site - 1][0],
                                    self.contraction_psi2[self.current_site - 1][1])
                psi, idx_psi = contra(psi_r, idx_r, self.contraction_psi2[self.current_site + 2][0],
                                self.contraction_psi2[self.current_site + 2][1])
                # Z, idz = contra(Z, idz, merg_l, idm_l)
        return psi

    def contraction_updat_twosite2(self):
        if self.going_righ==0:
            if self.current_site==self.n-2:
                merg,idm=contra(self.tensors[self.n-1],self.order[self.n-1],self.tensors[self.n-1],self.order_left[self.n-1])
                # self.contraction_z[self.current_site+1]=merg
                merg_psi, idm_psi = contra(self.image_tensors[self.n - 1], self.order_right[self.n - 1], self.tensors[self.n - 1],
                                   self.order_left[self.n - 1])

            else:


                merg, idm = contra(self.tensors[self.current_site+1], self.order_left[self.current_site+1],
                                   self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
                merg, idm = contra(merg, idm, self.tensors[self.current_site+1], self.order[self.current_site+1])

                merg_psi, idm_psi = contra(self.tensors[self.current_site+1], self.order_left[self.current_site+1],
                                   self.contraction_psi2[self.current_site+2][0],self.contraction_psi2[self.current_site+2][1])
                merg_psi, idm_psi = contra(merg_psi, idm_psi, self.image_tensors[self.current_site + 1], self.order_right[self.current_site + 1])

            self.contraction_psi2[self.current_site+1]=([merg_psi, idm_psi])
            self.contraction_z[self.current_site+1]=([merg,idm])

        if self.going_righ == 1:
                if self.current_site == 0:
                    merg, idm = contra(self.tensors[0], self.order[0], self.tensors[0],self.order_left[0])
                    # self.contraction_z[self.current_site+1]=merg
                    merg_psi, idm_psi = contra(self.image_tensors[0], self.order_right[0], self.tensors[0], self.order_left[0])



                else:

                    merg, idm = contra(self.tensors[self.current_site ], self.order_left[self.current_site ],
                                       self.contraction_z[self.current_site -1][0],self.contraction_z[self.current_site-1][1])
                    merg, idm = contra(merg, idm, self.tensors[self.current_site ],
                                       self.order[self.current_site ])

                    merg_psi, idm_psi = contra(self.tensors[self.current_site], self.order_left[self.current_site],
                                       self.contraction_psi2[self.current_site - 1][0],
                                       self.contraction_psi2[self.current_site - 1][1])
                    merg_psi, idm_psi = contra(merg_psi, idm_psi, self.image_tensors[self.current_site],
                                       self.order_right[self.current_site])


                    self.contraction_psi2[self.current_site ] = ([merg_psi, idm_psi])
                    self.contraction_z[self.current_site ] = ([merg,idm])



    def gradient_descent2(self):
        merged_idx2 = self.merged_idx[:]
        for k in range(len(self.merged_idx)):
            if self.merged_idx[k] % 2 == 1:
                merged_idx2[k] = merged_idx2[k] - 8 * self.n
        merge_psi,merge_psi_idx=contra(self.image_tensors[self.current_site],self.order_right[self.current_site],
                                       self.image_tensors[self.current_site+1], self.order_right[self.current_site+1])
        if self.current_site==0:


            dZ,dZ_idx=contra(self.merged_tensor,merged_idx2,self.contraction_z[self.current_site+2][0],self.contraction_z[self.current_site+2][1])
            dPsi, dPsi_idx = contra(merge_psi,merge_psi_idx, self.contraction_psi2[self.current_site + 2][0],
                                self.contraction_psi2[self.current_site + 2][1])


        else:
            if self.current_site==self.n-2:
                dZ, dZ_idx = contra(self.merged_tensor, merged_idx2, self.contraction_z[self.current_site - 1][0],
                                    self.contraction_z[self.current_site -1][1])
                dPsi, dPsi_idx = contra(merge_psi,merge_psi_idx, self.contraction_psi2[self.current_site - 1][0],
                                    self.contraction_psi2[self.current_site - 1][1])

            else:
                merg_Z,idz=contra(self.merged_tensor, merged_idx2, self.contraction_z[self.current_site - 1][0],
                                    self.contraction_z[self.current_site -1][1])
                dZ, dZ_idx = contra(merg_Z,idz, self.contraction_z[self.current_site + 2][0],
                                    self.contraction_z[self.current_site + 2][1])
                merg_Psi, idpsi = contra(merge_psi,merge_psi_idx, self.contraction_psi2[self.current_site - 1][0],
                                    self.contraction_psi2[self.current_site - 1][1])
                dPsi, dPsi_idx = contra(merg_Psi, idpsi, self.contraction_psi2[self.current_site + 2][0],
                                    self.contraction_psi2[self.current_site + 2][1])

        # for im1 in [1,2]:
        #     for im2 in [1,2]:
        #         dpsi=torch.zeros((dPsi[0][0]).size()).cuda()
        #         im=(self.image[self.current_site,:]==im1)*(self.image[self.current_site+1,:]==im2)
        #         for j in range(self.m):
        #             if im[j]==1:
        #                 dpsi=2*dPsi[0][j]/self.psi[j]+dpsi
        #             dmerge=tensor_slide(dmerge,self.merged_idx,[2-im1,2-im2],[self.current_site*2,self.current_site*2+2],dpsi,(dPsi[0][-1]))
        psi_1=1.0/self.psi
        dPsi, dPsi_idx=contra(dPsi, dPsi_idx,psi_1,[-1])
        dmerge=tensor_add(2*dPsi/self.m,dPsi_idx,-2*4*dZ/self.Z,dZ_idx)

        gnorm = torch.norm(dmerge) / 80
        if (gnorm < 1.0): #% & & self.bond_dims(self.current_bond) <= 50;
            dmerge = dmerge / gnorm;
        dmerge = tensor_add(self.merged_tensor,self.merged_idx , self.learning_rate * dmerge,dPsi_idx)


        return dmerge

    def gradient_descent25(self,j,k):
        self.current_site=j
        self.merged_tensor,self.merged_idx=contra(self.tensors[j],self.order_left[j],self.tensors[k],self.order_left[k])
        merged_idx2 = self.merged_idx[:]
        psi_merg,psi_merg_idx=contra(self.image_tensors[j],self.order_right[j],self.image_tensors[k],self.order_right[k])
        for l in range(len(self.merged_idx)):
            if self.merged_idx[l] % 2 == 1:
                merged_idx2[l] = merged_idx2[l] - 8 * self.n

        mer_l,mer_l_idx=contra(self.tensors[0],self.order_left[0],self.tensors[0],self.order[0])
        psi_l, psi_l_idx = contra(self.tensors[0], self.order_left[0], self.image_tensors[0], self.order_right[0])
        for i in range(1,j):
            mer_l,mer_l_idx=contra(mer_l,mer_l_idx,self.tensors[i],self.order_left[i])
            mer_l, mer_l_idx = contra(mer_l, mer_l_idx, self.tensors[i], self.order[i])

            psi_l, psi_l_idx = contra(psi_l, psi_l_idx, self.tensors[i], self.order_left[i])
            psi_l, psi_l_idx = contra(psi_l, psi_l_idx, self.image_tensors[i], self.order_right[i])

        mer_r, mer_r_idx = contra(self.tensors[self.n-1], self.order_left[self.n-1], self.tensors[self.n-1], self.order[self.n-1])

        psi_r, psi_r_idx = contra(self.tensors[self.n - 1], self.order_left[self.n - 1], self.image_tensors[self.n - 1],
                                  self.order_right[self.n - 1])
        for i in np.arange(self.n-2,k,-1):
            mer_r, mer_r_idx = contra(mer_r, mer_r_idx, self.tensors[i], self.order_left[i])
            mer_r, mer_r_idx = contra(mer_r, mer_r_idx, self.tensors[i], self.order[i])

            psi_r, psi_r_idx = contra(psi_r, psi_r_idx , self.tensors[i], self.order_left[i])
            psi_r, psi_r_idx = contra(psi_r, psi_r_idx , self.image_tensors[i], self.order_right[i])

        mer_m,mer_m_idx=contra(self.tensors[j+1],self.order_left[j+1],self.tensors[j+1],self.order[j+1])
        psi_m, psi_m_idx = contra(self.tensors[j + 1], self.order_left[j + 1], self.image_tensors[j + 1], self.order_right[j + 1])
        for i in range(j+2,k):
            mer_m, mer_m_idx = contra(mer_m, mer_m_idx, self.tensors[i], self.order_left[i])
            mer_m, mer_m_idx = contra(mer_m, mer_m_idx, self.tensors[i], self.order[i])

            psi_m, psi_m_idx = contra(psi_m, psi_m_idx, self.tensors[i], self.order_left[i])
            psi_m, psi_m_idx = contra(psi_m, psi_m_idx, self.image_tensors[i], self.order_right[i])

        dZ,dZ_idx=contra(mer_l,mer_l_idx,self.merged_tensor,merged_idx2)
        dZ,dZ_idx=contra(dZ,dZ_idx,mer_r,mer_r_idx)
        dZ,dZ_idx=contra(dZ,dZ_idx,mer_m,mer_m_idx)

        dpsi, dpsi_idx = contra(psi_l, psi_l_idx, psi_merg,psi_merg_idx)
        dpsi, dpsi_idx = contra(dpsi, dpsi_idx, psi_r, psi_r_idx)
        dpsi, dpsi_idx = contra(dpsi, dpsi_idx, psi_m, psi_m_idx)

        psi_1 = 1.0 / self.psi
        dpsi, dpsi_idx = contra(dpsi, dpsi_idx, psi_1, [-1])
        dmerge = tensor_add(2 * dpsi/self.m, dpsi_idx, -2 * 4 * dZ / self.Z, dZ_idx)

        gnorm = torch.norm(dmerge) / 80
        if (gnorm < 1.0):  # % & & self.bond_dims(self.current_bond) <= 50;
            dmerge = dmerge / gnorm;
        dmerge = tensor_add(self.merged_tensor,self.merged_idx , self.learning_rate * dmerge,dpsi_idx)


        return dmerge


    def train2(self,loops):
        for loop in range(loops):
            self.going_righ=0
            sk=0
            t1=time.time()
            for i in np.arange(self.n - 2, 0, -1):

                self.current_site=i
                self.merged_tensor,self.merged_idx=contra(self.tensors[self.current_site],self.order_left[self.current_site],
                                      self.tensors[self.current_site+1],self.order_left[self.current_site+1])
                self.Z=self.compute_Z()
                self.psi=self.compute_psi2()
                dmerge=self.gradient_descent2()
                # nll=0
                # for k in range(self.m):
                #     nll=nll+(torch.log(self.psi[k] *self.psi[k] /self.Z))
                # nll=nll/self.m
                # print nll,i
                self.tensors[self.current_site], self.tensors[self.current_site + 1] = svd_update(
                    self.tensors[self.current_site], self.order[self.current_site],
                    self.tensors[self.current_site + 1], self.order[self.current_site + 1], dmerge,
                    self.going_righ, 1e-6, self.max_bond
                )
                self.contraction_updat_twosite2()

            self.going_righ = 1
            for i in range(self.n - 1):
                self.current_site = i
                self.merged_tensor, self.merged_idx = contra(self.tensors[self.current_site],
                                                             self.order_left[self.current_site],
                                                             self.tensors[self.current_site + 1],
                                                             self.order_left[self.current_site + 1])
                # t1 = time.time()
                self.Z = self.compute_Z()
                # t2=time.time()
                self.psi = self.compute_psi2()
                # nll = 0
                # for k in range(self.m):
                #     nll = nll + ( torch.log(self.psi[k] *self.psi[k]  / self.Z))
                # nll=nll/self.m
                # print nll,i
                # t3=time.time()
                dmerge = self.gradient_descent2()
                # t4=time.time()
                self.tensors[self.current_site], self.tensors[self.current_site + 1] = svd_update(
                    self.tensors[self.current_site], self.order[self.current_site],
                    self.tensors[self.current_site + 1], self.order[self.current_site + 1], dmerge,
                    self.going_righ, 1e-6, self.max_bond
                )
                # t5=time.time()
                self.contraction_updat_twosite2()
                # t6=time.time()
                # print t2-t1,t3-t2,t4-t3,t5-t4,t6-t5
                if i == 400:
                    print self.m

            for j in range(len(self.links)):
                self.Z = self.compute_Z()
                self.psi = self.compute_psi2()
                k0 = self.links[j][0]
                k1=self.links[j][1]
                dmerge=self.gradient_descent25(k0,k1)
                self.tensors[k0], self.tensors[k1] = svd_update(
                    self.tensors[k0], self.order[k0],
                    self.tensors[k1], self.order[k1], dmerge,
                    self.going_righ, 2e-6, self.max_bond
                )
                self.contraction_update_all_left2()
            nll=0
            for k in range(self.m):
                nll = nll + (torch.log(self.psi[k] *self.psi[k] / self.Z))
            nll = nll / self.m
            nll=nll.float()
            print nll
            # if nll>-30 & sk==0:
            #     sk=1
            #     for i in range(self.n):
            #         self.tensors[i]=self.tensors[i].float()
            #         self.image_tensors[i]=self.image_tensors[i].float()
            #         self.contraction_psi2[i][0]=self.contraction_psi2[i][0].float()
            #         self.contraction_z[i][0]=self.contraction_z[i][0].float()
            if nll>self.nll_history[-1]+2e-5:
                self.nll_history.append(nll)
            else:
                break
            t2=time.time()
            print t2-t1


data=sio.loadmat('mnist_100_images.mat')
images=data['train_x_binary']
images=torch.Tensor([images])
images=images.view([784,100])
images=images[:,0:20]
del data

# images=torch.ones([700,100])
# images[images>0.7]=2
# images[images<=0.7]=1
# net=tensor_net(T,0.0001,np.array([]),20,2)
# net.init_tensors()
# net.init_image_tensor()
# net.contraction_update_all_left()
# t1=time.time()
# net.train(50)
# t2=time.time()
net=tensor_net(images,0.001,np.array([[250,500]]),30,2)
net.init_tensors()
net.init_image_tensor()
net.contraction_update_all_left2()
t3=time.time()
net.train2(7)
t4=time.time()
print t4-t3
# print a