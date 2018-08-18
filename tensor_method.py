import numpy as np
import torch
import time


def contra(A, idxa, B, idxb):
    # Ma = np.zeros([3, len(idxa)], dtype=int)
    # Ma[0, :] = np.array(A.size())
    # Ma[1, :] = np.array(idxa)
    # Ma[2, :] = np.arange(0, len(idxa), 1, int)
    # Mb = np.zeros([3, len(idxb)], dtype=int)
    # Mb[0, :] = np.array(B.size())
    # Mb[1, :] = np.array(idxb)
    # Mb[2, :] = np.arange(0, len(idxb), 1, int)
    sa=np.array(A.size())
    sb=np.array(B.size())
    i = 0
    p = 1
    ran_a=range(len(idxa))
    ran_b=range(len(idxb))
    ra=range(len(idxa))
    rb=range(len(idxb))
    idxa_1=[]
    idxb_1=[]

    for ia in ra:
        if sum(np.array(idxb)==idxa[ia])!=0:
            ran_a.remove(ia)
            ran_a=list([ia])+ran_a
            ran_b.remove(idxb.index(idxa[ia]))
            ran_b=list([idxb.index(idxa[ia])])+ran_b
            i=i+1
            p=p*sa[ia]
    idxa_1=np.array(idxa)[ran_a]
    idxb_1=np.array(idxb)[ran_b]
    sa=sa[ran_a]
    sb=sb[ran_b]

    tensor_A = A.permute(ran_a).contiguous()
    tensor_B = B.permute(ran_b).contiguous()
    tensor_A = tensor_A.view(p, -1)
    tensor_B = tensor_B.view(p, -1)
    tensor_C = torch.mm(tensor_A.t(), tensor_B)
    # Ma[:, :] = Ma[:, -1::-1]

    if i == 0:
        C = tensor_C.view(list(sa) + list(sb))
        idxc = list(idxa_1) + list(idxb_1)

    else:

        C = tensor_C.view(list(sa[ i:]) + list(sb[i:]))
        idxc = list(idxa_1[ i:]) + list(idxb_1[i:])
    return C, idxc


def svd_update(A,idxa,B,idxb,T_C,going_right,err,max_bond):
    sa = np.array(A.size())
    sb = np.array(B.size())
    i = 0
    p = 1
    ran_a = range(len(idxa))
    ran_b = range(len(idxb))
    ra = range(len(idxa))
    rb = range(len(idxb))
    idxa_1 = []
    idxb_1 = []

    for ia in ra:
        if sum(np.array(idxb) == idxa[ia]) != 0:
            ran_a.remove(ia)
            ran_a = list([ia]) + ran_a
            ran_b.remove(idxb.index(idxa[ia]))
            ran_b = list([idxb.index(idxa[ia])]) + ran_b
            i = i + 1
            p = p * sa[ia]
    idxa_1 = np.array(idxa)[ran_a]
    idxb_1 = np.array(idxb)[ran_b]
    sa = sa[ran_a]
    sb = sb[ran_b]
    if torch.is_tensor(T_C):
        c=1
        for k in range(len(sa[i:])):
            c=c*sa[i+k]
        T_C=T_C.view(c,-1)
        tensor_C=T_C
    else:
        tensor_A = A.permute(ran_a).contiguous()
        tensor_B = B.permute(ran_b).contiguous()
        tensor_A = tensor_A.view(p, -1)
        tensor_B = tensor_B.view(p, -1)
        tensor_C = torch.mm(tensor_A.t(), tensor_B)

    # Ma[:, :] = Ma[:, -1::-1]
    U,S,V=torch.svd(tensor_C)
    bond=(sum(S>=S[0]*err))
    bond=bond.item()
    bond=min(bond,max_bond)
    U=U[:,:bond]
    # U=U.t()
    S=S[:bond]
    V=V.t()
    V=V[:bond,:]
    # print(torch.mm(U,U.t()))
    # print(torch.mm(V,V.t()))
    if going_right==1:
        V=torch.mm(S*torch.eye(bond,dtype=V.dtype),V)
        V=V / torch.sqrt(sum(sum(V*V)))
    else:
        U=torch.mm(U,torch.eye(bond,dtype=V.dtype)*S)
        U = U / torch.sqrt(sum(sum(U * U)))

    # for i in np.arange(bond/2+1,1,-1):
    #     if bond%i ==0:
    #        bond_1=i
    #        bond_2=bond/i

    ra_1=np.zeros(len(idxa),dtype=int)
    rb_1 = np.zeros(len(idxb),dtype=int)
    ra_1[ran_a]=range(len(idxa))
    rb_1[ran_b] = range(len(idxb))

    if i==0:
        tensor_U=U.t().view(sa)
        tensor_V = V.view(sb)

    else:
        tensor_U = U.t().view([bond]+list(sa[i:]))
        tensor_V = V.view([bond]+list(sb[i:]))
    tensor_U = tensor_U.permute(list(ra_1))
    tensor_V = tensor_V.permute(list(rb_1))
    return tensor_U,tensor_V

def qr_update(A,idxa,B,idxb,T_C,going_right,err,max_bond):
    sa = np.array(A.size())
    sb = np.array(B.size())
    i = 0
    p = 1
    ran_a = range(len(idxa))
    ran_b = range(len(idxb))
    ra = range(len(idxa))
    rb = range(len(idxb))
    idxa_1 = []
    idxb_1 = []

    for ia in ra:
        if sum(np.array(idxb) == idxa[ia]) != 0:
            ran_a.remove(ia)
            ran_a = list([ia]) + ran_a
            ran_b.remove(idxb.index(idxa[ia]))
            ran_b = list([idxb.index(idxa[ia])]) + ran_b
            i = i + 1
            p = p * sa[ia]
    idxa_1 = np.array(idxa)[ran_a]
    idxb_1 = np.array(idxb)[ran_b]
    sa = sa[ran_a]
    sb = sb[ran_b]

    tensor_A = A.permute(ran_a).contiguous()
    tensor_B = B.permute(ran_b).contiguous()
    tensor_A = tensor_A.view(p, -1)
    tensor_B = tensor_B.view(p, -1)
    tensor_C = torch.mm(tensor_A.t(), tensor_B)
    if torch.is_tensor(T_C):
        T_C=T_C.view_as(tensor_C)
        tensor_C=T_C
    # Ma[:, :] = Ma[:, -1::-1]
    U,S,V=torch.svd(tensor_C)
    bond=(sum(S>S[0]*err))
    bond=bond.item()
    bond=min(bond,max_bond)
    U=U[:,:bond]
    # U=U.t()
    S=S[:bond]
    V=V.t()
    V=V[:bond,:]
    # print(torch.mm(U,U.t()))
    # print(torch.mm(V,V.t()))
    if going_right==1:
        V=torch.mm(S*torch.eye(bond).cuda(),V)
        V=V / torch.sqrt(sum(sum(V*V)))
    else:
        U=torch.mm(U,torch.eye(bond).cuda()*S)
        U = U / torch.sqrt(sum(sum(U * U)))

    # for i in np.arange(bond/2+1,1,-1):
    #     if bond%i ==0:
    #        bond_1=i
    #        bond_2=bond/i

    ra_1=np.zeros(len(idxa),dtype=int)
    rb_1 = np.zeros(len(idxb),dtype=int)
    ra_1[ran_a]=range(len(idxa))
    rb_1[ran_b] = range(len(idxb))

    if i==0:
        tensor_U=U.t().view(sa)
        tensor_V = V.view(sb)

    else:
        tensor_U = U.t().view([bond]+list(sa[i:]))
        tensor_V = V.view([bond]+list(sb[i:]))
    tensor_U = tensor_U.permute(list(ra_1))
    tensor_V = tensor_V.permute(list(rb_1))
    return tensor_U,tensor_V


def tensor_add(A,idxa,B,idxb):
    rb_1=[]
    for i in range(len(idxa)):
        rb_1.append(idxb.index(idxa[i]))
    B=B.permute(list(rb_1))
    C=A+B
    return C


def tensor_slide(A,idxa,v,idxv,C,idxc):
    b=[]
    d=[]
    for i in range(len(idxa)):
        if sum(torch.Tensor(idxv)==idxa[i])==0:
            b.append(i)
        else:
            d.append(i)
    ra=d+b
    A=A.permute(ra)
    sa=A.size()[2:]
    idxa2=np.array(idxa)[ra]
    midA=A[v[0],v[1]].view(sa)
    A[v[0],v[1]]=tensor_add(midA,idxa2[2:],C,idxc)
    ra_1 = np.zeros(len(idxa), dtype=int)
    ra_1[ra] = range(len(idxa))
    ra_1=list(ra_1)
    A=A.permute((ra_1))
    return A

# def load_mnist(file):
#     mat = scipy.io.loadmat(file)
#     return mat

# a=torch.rand([1,10],dtype=torch.float64)
# b=a.float()
# print b