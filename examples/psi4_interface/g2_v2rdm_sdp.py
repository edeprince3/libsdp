"""
the semidefinite program for variational two-electron reduced density matrix theory
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

class g2_v2rdm_sdp():

    def __init__(self, nalpha, nbeta, nmo, oei, tei, q2 = False, g2 = True, constrain_spin = False, t2 = False):
        """
        SDP problem

        :param nalpha: number of alpha electrons
        :param nbeta:  number of beta electrons
        :param nmo:    number of spatial molecular orbitals
        :param oei:    core Hamiltonian matrix
        :param tei:    two-electron repulsion integrals

        members:

        b:             the constraint vector
        F:             list of rows of constraint matrix in sparse format; 
                       note that F[0] is actually the vector defining the 
                       problem (contains the one- and two-electron integrals)
        dimensions:    list of dimensions of blocks of primal solution
        offsets:       list of offsets of blocks of primal solution
        block_id:      list of block ids for blocks of primal solution
        nmo:           number of spatial molecular orbitals
        nalpha:        number of alpha electrons
        nbeta:         number of beta electrons
        bas_ab:        orbital pairs to geminals map (ab-type)
        bas_aa:        orbital pairs to geminals map (aa-type)
        ibas_ab:       geminals to orbital pairs map (ab-type)
        ibas_aa:       geminals to orbital pairs map (aa-type)
        """

        # nmo, nalpha, nbeta
        self.nmo = nmo
        self.nalpha = nalpha
        self.nbeta = nbeta

        # orbital to geminal maps
        self.bas_aa = []
        self.bas_ab = []

        # geminal to orbital maps
        self.ibas_aa = np.zeros((nmo, nmo), dtype='int32')
        self.ibas_ab = np.zeros((nmo, nmo), dtype='int32')

        count_aa = 0
        count_ab = 0
        for i in range (0, nmo):
            for j in range (0, nmo):
                self.bas_ab.append([i, j])
                self.ibas_ab[i, j] = count_ab
                count_ab += 1
                if i < j:
                    self.bas_aa.append([i, j])
                    self.ibas_aa[i, j] = count_aa
                    self.ibas_aa[j, i] = count_aa
                    count_aa += 1
                elif i == j:
                    self.ibas_aa[i, j] = -999

        # block dimensions
        self.dimensions = []
        self.dimensions.append(nmo) # d1a
        self.dimensions.append(nmo) # d1b
        #self.dimensions.append(nmo*nmo) # d2ab
        #self.dimensions.append(nmo*(nmo-1)//2) # d2aa
        #self.dimensions.append(nmo*(nmo-1)//2) # d2bb
        self.dimensions.append(nmo) # q1a
        self.dimensions.append(nmo) # q1b

        if q2: 
            self.dimensions.append(nmo*nmo) # q2ab
            self.dimensions.append(nmo*(nmo-1)//2) # q2aa
            self.dimensions.append(nmo*(nmo-1)//2) # q2bb

        if g2: 
            self.dimensions.append(2*nmo*nmo) # g2aaaa / g2aabb / g2bbaa/ g2bbbb
            self.dimensions.append(nmo*nmo) # g2ab
            self.dimensions.append(nmo*nmo) # g2ba

        # block ids ... block zero defines the objective function
        # this is the only dangerous part ... need to be sure the order of the ids matches the dimensions above

        #blocks = ['d1a', 'd1b', 'd2ab', 'd2aa', 'd2bb', 'q1a', 'q1b']
        blocks = ['d1a', 'd1b', 'q1a', 'q1b']

        if q2 :
            blocks.append('q2ab')
            blocks.append('q2aa')
            blocks.append('q2bb')

        if g2 :
            blocks.append('g2aa')
            blocks.append('g2ab')
            blocks.append('g2ba')

        self.block_id = {
        }
        count = 1
        for block in blocks:
            self.block_id[block] = count
            count = count + 1
    
        # block offsets
        self.offsets = {
        }
        self.offsets[blocks[0]] = 0
        for i in range (1, len(blocks)):
            self.offsets[blocks[i]] = self.offsets[blocks[i-1]] + self.dimensions[self.block_id[blocks[i-1]]]**2
    
        # number of blocks
        nblocks = len(self.dimensions)
    
        # F0  ... the integrals
        block_number=[]
        row=[]
        column=[]
        value=[]
    
        F = libsdp.sdp_matrix()

        dum = np.einsum('ijkk->ij', tei)
        for i in range (0, nmo):
            for j in range (0, nmo):
                block_number.append(self.block_id['d1a'])
                row.append(i+1)
                column.append(j+1)
                value.append(oei[i][j] + dum[i][j])
    
        for i in range (0, nmo):
            for j in range (0, nmo):
                block_number.append(self.block_id['d1b'])
                row.append(i+1)
                column.append(j+1)
                value.append(oei[i][j] + dum[i][j])
    
        # (ik|jl) aaaa -> g2aaaa
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                block_number.append(self.block_id['g2aa'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-0.5 * tei[i][k][j][l])
    
        # (ik|jl) bbbb -> g2bbbb
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                block_number.append(self.block_id['g2aa'])
                row.append(ij+len(self.bas_ab)+1)
                column.append(kl+len(self.bas_ab)+1)
                value.append(-0.5 * tei[i][k][j][l])
    
        # (ik|jl) aabb -> g2abab
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                block_number.append(self.block_id['g2ab'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-0.5 * tei[i][k][j][l])
    
        # (ik|jl) bbaa -> g2baba
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                block_number.append(self.block_id['g2ba'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-0.5 * tei[i][k][j][l])
    
        F.block_number = block_number
        F.row          = row
        F.column       = column
        F.value        = value

        self.F = []
        self.F.append(F)
        
        # build constraints (F1, F2, ...)
    
        self.b = []

        # trace of d1a
        self.trace_d1(self.block_id['d1a'], nalpha)
    
        # trace of d1b
        self.trace_d1(self.block_id['d1b'], nbeta)

        # trace of d2ab
        #self.trace_d2(self.block_id['d2ab'], self.bas_ab, nalpha*nbeta)

        # trace of d2aa
        #F = self.trace_d2(self.block_id['d2aa'], self.bas_aa, nalpha*(nalpha-1)/2)

        # trace of d2bb
        #F = self.trace_d2(self.block_id['d2bb'], self.bas_aa, nbeta*(nbeta-1)/2)

        # antisymmetry of d2aa via g2aaaa
        self.g2aaaa_antisymmetry(0, self.block_id['d1a'])

        # antisymmetry of d2aa via g2aaaa
        self.g2aaaa_antisymmetry(len(self.bas_ab), self.block_id['d1b'])

        # g2aaaa -> d1a ... 4 contractions
        self.contract_g2aaaa_d1a(0, self.block_id['d1a'], self.nalpha)

        # g2bbbb -> d1b ... 4 contractions
        self.contract_g2aaaa_d1a(len(self.bas_ab), self.block_id['d1b'], self.nbeta)

        # g2aabb -> d1a, d1b ... 2 contractions
        self.contract_g2aabb_d1(0, len(self.bas_ab), self.nbeta, self.block_id['d1a'], self.nalpha, self.block_id['d1b'])

        # g2bbaa -> d1a, d1b ... 2 contractions
        self.contract_g2aabb_d1(len(self.bas_ab), 0, self.nalpha, self.block_id['d1b'], self.nbeta, self.block_id['d1a'])

        # g2abab -> d1a, d1b ... 2 contractions
        self.contract_g2abab_d1(self.block_id['g2ab'], self.nbeta, self.block_id['d1a'], self.nalpha, self.block_id['d1b'])

        # g2baba -> d1a, d1b ... 2 contractions
        self.contract_g2abab_d1(self.block_id['g2ba'], self.nalpha, self.block_id['d1b'], self.nbeta, self.block_id['d1a'])

        # trace of g2aaaa
        self.trace_g2(0, self.block_id['g2aa'], self.nalpha * self.nmo - self.nalpha * (self.nalpha - 1.0))

        # trace of g2bbbb
        self.trace_g2(len(self.bas_ab), self.block_id['g2aa'], self.nbeta * self.nmo - self.nbeta * (self.nbeta - 1.0))

        # trace of g2abab
        self.trace_g2(0, self.block_id['g2ab'], self.nalpha * self.nmo - self.nalpha * self.nbeta)

        # trace of g2baba
        self.trace_g2(0, self.block_id['g2ba'], self.nbeta * self.nmo - self.nalpha * self.nbeta)

        # d2ab -> d1a
        #self.contract_d2ab_d1a()

        # d2ab -> d1b
        #self.contract_d2ab_d1b()

        # d2aa -> d1a
        #self.contract_d2aa_d1a(self.block_id['d2aa'], self.block_id['d1a'], nalpha)

        # d2bb -> d1b
        #self.contract_d2aa_d1a(self.block_id['d2bb'], self.block_id['d1b'], nbeta)

        # d1a <-> q1a
        self.d1_q1_mapping(self.block_id['d1a'], self.block_id['q1a'])

        # d1b <-> q1b
        self.d1_q1_mapping(self.block_id['d1b'], self.block_id['q1b'])

        #if q2: 

        #    # q2ab <-> d2ab, d1a, d1b
        #    self.q2ab_mapping()

        #    # q2aa <-> d2aa, d1a
        #    self.q2aa_mapping(self.block_id['d2aa'], self.block_id['d1a'], self.block_id['q2aa'])

        #    # q2bb <-> d2bb, d1b
        #    self.q2aa_mapping(self.block_id['d2bb'], self.block_id['d1b'], self.block_id['q2bb'])

        #if g2:

        #    # g2aaaa/aabb/bbaa/bbbb <-> d2ab, d2aa, d2bb, d1a, d1b
        #    self.g2aa_mapping()

        #    # g2ab <-> d2ab, d1a
        #    self.g2ab_mapping()

        #    # g2ba <-> d2ab, d1b
        #    self.g2ba_mapping()

        #if constrain_spin :

        #    # <s^2> 
        #    self.constrain_s2()

        #    # spin block structure (currently only for singlets, D1a = D1b, D2aa(ijkl) = D2ab(ijkl) - D2ab(ijlk), etc.)
        #    if self.nalpha == self.nbeta:
        #        self.constrain_spin_block_structure()

    def trace_d1(self, block_id, n):
        """
        tr(d1) = n

        :param block_id: the block id for this block of d1
        :param n:        the number of electrons
        """

        # Tr(D1)
        block_number=[]
        row=[]
        column=[]
        value=[]
    
        for i in range (0, self.nmo):
            block_number.append(block_id)
            row.append(i+1)
            column.append(i+1)
            value.append(1.0)
    
        F = libsdp.sdp_matrix()
        F.block_number = block_number
        F.row          = row
        F.column       = column
        F.value        = value
    
        self.F.append(F)
        self.b.append(n)

    def trace_d2(self, block_id, bas, n):
        """
        tr(d2) = n 

        :param bas:      the list of geminals (ab, aa, or bb)
        :param block_id: the relevant block of d2 (ab, aa, or bb)
        :param n:        the trace
        """
    
        # Tr(D2)
        block_number=[]
        row=[]
        column=[]
        value=[]
   
        for ij in range (0, len(bas)):
            block_number.append(block_id)
            row.append(ij+1)
            column.append(ij+1)
            value.append(1.0)
   
        F = libsdp.sdp_matrix()
        F.block_number = block_number
        F.row          = row
        F.column       = column
        F.value        = value

        self.F.append(F)
        self.b.append(n)

    def contract_d2ab_d1a(self):
        """
        contract d2ab to d1a
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for k in range (0, self.nmo):
   
                    ik = self.ibas_ab[i, k]
                    jk = self.ibas_ab[j, k]
                    block_number.append(self.block_id['d2ab'])
                    row.append(ik+1)
                    column.append(jk+1)
                    value.append(1.0)
   
                block_number.append(self.block_id['d1a'])
                row.append(i+1)
                column.append(j+1)
                value.append(-self.nbeta)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def contract_d2ab_d1b(self):
        """
        contract d2ab to d1b
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for k in range (0, self.nmo):
   
                    ki = self.ibas_ab[k, i]
                    kj = self.ibas_ab[k, j]
                    block_number.append(self.block_id['d2ab'])
                    row.append(ki+1)
                    column.append(kj+1)
                    value.append(1.0)
   
                block_number.append(self.block_id['d1b'])
                row.append(i+1)
                column.append(j+1)
                value.append(-self.nalpha)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def contract_d2aa_d1a(self, d2_block_id, d1_block_id, n):
        """
        contract d2aa to d1a (or d2bb to d1b)

        :param d2_block_id: the block id for d2aa or d2bb
        :param d1_block_id: the block id for d1a or d1b
        :param n: the number of alpha or beta electrons
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for k in range (0, self.nmo):
   
                    if i == k or j == k :
                        continue

                    ik = self.ibas_aa[i, k]
                    jk = self.ibas_aa[j, k]
                    block_number.append(d2_block_id)
                    row.append(ik+1)
                    column.append(jk+1)
                    sg = 1
                    if i > k :
                        sg = -sg
                    if j > k :
                        sg = -sg
                    value.append(sg * 1.0)
   
                block_number.append(d1_block_id)
                row.append(i+1)
                column.append(j+1)
                value.append(-(n-1))

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def d1_q1_mapping(self, d1_block_id, q1_block_id):
        """
        map d1a to q1a (or d1b to q1b)

        :param d1_block_id: the block id for d1a or d1b
        :param q1_block_id: the block id for q1a or q1b
        """

        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                block_number.append(d1_block_id)
                row.append(i+1)
                column.append(j+1)
                value.append(1.0)

                block_number.append(q1_block_id)
                row.append(j+1)
                column.append(i+1)
                value.append(1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(delta[i, j])
                self.F.append(myF)

    def q2aa_mapping(self, d2_block_id, d1_block_id, q2_block_id):
        """
        map q2aa to d2aa and d1a (or q2bb to d2bb and d1b)

        :param d2_block_id: the block id for d2aa or d2bb
        :param d1_block_id: the block id for d1a or d1b
        :param q2_block_id: the block id for q1a or q1b

        - d(j,l) d(i,k) 
        + d(i,l) d(j,k) 

        =

        - k* i d(j,l) 
        + k* j d(i,l) 
        + l* i d(j,k) 
        - l* j d(i,k) 
        + k* l* j i 
        - i j l* k* 
        """

        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        for ij in range (0, len(self.bas_aa)):
            i = self.bas_aa[ij][0]
            j = self.bas_aa[ij][1]
            for kl in range (0, len(self.bas_aa)):
                k = self.bas_aa[kl][0]
                l = self.bas_aa[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # + k* l* j i
                block_number.append(d2_block_id)
                row.append(kl+1)
                column.append(ij+1)
                value.append(1.0)

                # - i j l* k*
                block_number.append(q2_block_id)
                row.append(ij+1)
                column.append(kl+1)
                value.append(-1.0)

                # - k* i d(j,l)
                if j == l:
                    block_number.append(d1_block_id)
                    row.append(k+1)
                    column.append(i+1)
                    value.append(-1.0)

                # + k* j d(i,l)
                if i == l:
                    block_number.append(d1_block_id)
                    row.append(k+1)
                    column.append(j+1)
                    value.append(1.0)

                # + l* i d(j,k)
                if j == k:
                    block_number.append(d1_block_id)
                    row.append(l+1)
                    column.append(i+1)
                    value.append(1.0)

                # - l* j d(i,k)
                if i == k:
                    block_number.append(d1_block_id)
                    row.append(l+1)
                    column.append(j+1)
                    value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(- delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k])
                self.F.append(myF)

    def q2ab_mapping(self):
        """
        map q2ab to d2ab and d1a and d1b

        - d(j,l) d(i,k) 

        =

        - k* i d(j,l) 
        - l* j d(i,k) 
        + k* l* j i 
        - i j l* k* 
        """

        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # + k* l* j i
                block_number.append(self.block_id['d2ab'])
                row.append(kl+1)
                column.append(ij+1)
                value.append(1.0)

                # - i j l* k*
                block_number.append(self.block_id['q2ab'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-1.0)

                # - k* i d(j,l)
                if j == l:
                    block_number.append(self.block_id['d1a'])
                    row.append(k+1)
                    column.append(i+1)
                    value.append(-1.0)

                # - l* j d(i,k)
                if i == k:
                    block_number.append(self.block_id['d1b'])
                    row.append(l+1)
                    column.append(j+1)
                    value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                # - d(j,l) d(i,k) 
                self.b.append(- delta[i, k] * delta[j, l])
                self.F.append(myF)

    def g2ab_mapping(self):
        """
        map g2ab to d2ab and d1a

        0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
           
        """

        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # - i* l* j k
                il = self.ibas_ab[i, l]
                kj = self.ibas_ab[k, j]
                block_number.append(self.block_id['d2ab'])
                row.append(il+1)
                column.append(kj+1)
                value.append(-1.0)

                # - i* j l* k 
                block_number.append(self.block_id['g2ab'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    block_number.append(self.block_id['d1a'])
                    row.append(i+1)
                    column.append(k+1)
                    value.append(1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def g2ba_mapping(self):
        """
        map g2ba to d2ab and d1b

        0 = - l* i* k j  - i* j l* k  + i* k d(j,l) 
           
        """

        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # - l* i* k j
                li = self.ibas_ab[l, i]
                jk = self.ibas_ab[j, k]
                block_number.append(self.block_id['d2ab'])
                row.append(li+1)
                column.append(jk+1)
                value.append(-1.0)

                # - i* j l* k 
                block_number.append(self.block_id['g2ba'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    block_number.append(self.block_id['d1b'])
                    row.append(i+1)
                    column.append(k+1)
                    value.append(1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def g2aa_mapping(self):
        """
        map g2aaaa/aabb/bbaa/bbbb to d2ab, d2aa, d2bb, d1a, and d1b

        aaaa:
        0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 

        aabb:
        0 = + i* l* k j  - i* j l* k  

        bbaa:
        0 = + i* l* k j  - i* j l* k  

        bbbb:
        0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
           
        """

        return

        # aaaa:
        # 0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # - i* l* j k
                if i != l and k != j:
                    il = self.ibas_aa[i][l]
                    kj = self.ibas_aa[k][j]
                    block_number.append(self.block_id['d2aa'])
                    row.append(il+1)
                    column.append(kj+1)
                    sg = 1
                    if i > l:
                        sg = -sg
                    if k > j:
                        sg = -sg
                    value.append(-sg)

                # - i* j l* k 
                block_number.append(self.block_id['g2aa'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    block_number.append(self.block_id['d1a'])
                    row.append(i+1)
                    column.append(k+1)
                    value.append(1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        # bbbb:
        # 0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # - i* l* j k
                if i != l and k != j:
                    il = self.ibas_aa[i][l]
                    kj = self.ibas_aa[k][j]
                    block_number.append(self.block_id['d2bb'])
                    row.append(il+1)
                    column.append(kj+1)
                    sg = 1
                    if i > l:
                        sg = -sg
                    if k > j:
                        sg = -sg
                    value.append(-sg)

                # - i* j l* k 
                block_number.append(self.block_id['g2aa'])
                row.append(len(self.bas_ab) + ij + 1)
                column.append(len(self.bas_ab) + kl + 1)
                value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    block_number.append(self.block_id['d1b'])
                    row.append(i+1)
                    column.append(k+1)
                    value.append(1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        # aabb:
        # 0 = + i* l* k j  - i* j l* k
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # + i* l* j k
                il = self.ibas_ab[i, l]
                jk = self.ibas_ab[j, k]
                block_number.append(self.block_id['d2ab'])
                row.append(il+1)
                column.append(jk+1)
                value.append(1.0)

                # - i* j l* k 
                block_number.append(self.block_id['g2aa'])
                row.append(ij + 1)
                column.append(len(self.bas_ab) + kl + 1)
                value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        # bbaa:
        # 0 = + l* i* j k  - i* j l* k
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                # + l* i* k j
                li = self.ibas_ab[l, i]
                kj = self.ibas_ab[k, j]
                block_number.append(self.block_id['d2ab'])
                row.append(li+1)
                column.append(kj+1)
                value.append(1.0)

                # - i* j l* k 
                block_number.append(self.block_id['g2aa'])
                row.append(len(self.bas_ab) + ij + 1)
                column.append(kl + 1)
                value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def constrain_s2(self):
        """
        constrain <s^2>
        """

        ms = 0.5 * (self.nalpha - self.nbeta)

        block_number=[]
        row=[]
        column=[]
        value=[]

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                ij = self.ibas_ab[i, j]
                ji = self.ibas_ab[j, i]

                block_number.append(self.block_id['d2ab'])
                row.append(ij+1)
                column.append(ji+1)
                value.append(1.0)

        myF = libsdp.sdp_matrix()
        myF.block_number = block_number
        myF.row          = row
        myF.column       = column
        myF.value        = value

        self.b.append(0.5 * (self.nalpha + self.nbeta) + ms*ms - ms*(ms+1.0))
        self.F.append(myF)

    def constrain_spin_block_structure(self):
        """
        constrain spin-block structure

        for singlets:     d1a = d1b
                          d2ab(ijkl) = d2ab(jilk)
                          d2aa(ijkl) = 1/2 ( d2ab(ijkl) - d2ab(ijlk) - d2ab(jikl) + d2ab(jilk) )
                          d2bb(ijkl) = 1/2 ( d2ab(ijkl) - d2ab(ijlk) - d2ab(jikl) + d2ab(jilk) )

                          TODO: see J. Chem. Phys. 136, 014110 (2012)
                          d200(ijkl) = 1/(2 sqrt(1+dij)sqrt(1+dkl) ) ( d2ab(ijkl) + d2ab(ijlk) + d2ab(jikl) + d2ab(jilk) )

        for non-singlets: TODO: see J. Chem. Phys. 136, 014110 (2012) 
        """

        # d1a = d1b
        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]

                block_number.append(self.block_id['d1a'])
                row.append(i+1)
                column.append(j+1)
                value.append(1.0)

                block_number.append(self.block_id['d1b'])
                row.append(i+1)
                column.append(j+1)
                value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value

                self.b.append(0.0)
                self.F.append(myF)

        # d2ab = d2ba
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            ji = self.ibas_ab[j, i]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                lk = self.ibas_ab[l, k]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                block_number.append(self.block_id['d2ab'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(1.0)

                block_number.append(self.block_id['d2ab'])
                row.append(ji+1)
                column.append(lk+1)
                value.append(-1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value

                self.b.append(0.0)
                self.F.append(myF)

        # d2aa(ijkl) = 1/2 ( d2ab(ijkl) - d2ab(ijlk) - d2ab(jikl) + d2ab(jilk) )
        for ij in range (0, len(self.bas_aa)):
            i = self.bas_aa[ij][0]
            j = self.bas_aa[ij][1]
            ij_ab = self.ibas_ab[i, j]
            ji_ab = self.ibas_ab[j, i]
            for kl in range (0, len(self.bas_aa)):
                k = self.bas_aa[kl][0]
                l = self.bas_aa[kl][1]
                kl_ab = self.ibas_ab[k, l]
                lk_ab = self.ibas_ab[l, k]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                block_number.append(self.block_id['d2aa'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(1.0)

                block_number.append(self.block_id['d2ab'])
                row.append(ij_ab+1)
                column.append(kl_ab+1)
                value.append(-0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ij_ab+1)
                column.append(lk_ab+1)
                value.append(0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ji_ab+1)
                column.append(kl_ab+1)
                value.append(0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ji_ab+1)
                column.append(lk_ab+1)
                value.append(-0.5)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value

                self.b.append(0.0)
                self.F.append(myF)

        # d2bb(ijkl) = 1/2 ( d2ab(ijkl) - d2ab(ijlk) - d2ab(jikl) + d2ab(jilk) )
        for ij in range (0, len(self.bas_aa)):
            i = self.bas_aa[ij][0]
            j = self.bas_aa[ij][1]
            ij_ab = self.ibas_ab[i, j]
            ji_ab = self.ibas_ab[j, i]
            for kl in range (0, len(self.bas_aa)):
                k = self.bas_aa[kl][0]
                l = self.bas_aa[kl][1]
                kl_ab = self.ibas_ab[k, l]
                lk_ab = self.ibas_ab[l, k]
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                block_number.append(self.block_id['d2bb'])
                row.append(ij+1)
                column.append(kl+1)
                value.append(1.0)

                block_number.append(self.block_id['d2ab'])
                row.append(ij_ab+1)
                column.append(kl_ab+1)
                value.append(-0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ij_ab+1)
                column.append(lk_ab+1)
                value.append(0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ji_ab+1)
                column.append(kl_ab+1)
                value.append(0.5)

                block_number.append(self.block_id['d2ab'])
                row.append(ji_ab+1)
                column.append(lk_ab+1)
                value.append(-0.5)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value

                self.b.append(0.0)
                self.F.append(myF)

    def contract_g2aaaa_d1a(self, offset, d1_block_id, n):
        """
        contract g2aaaa to d1a (or g2bbbb to d1b) ... there are 4 different contractions

        :param offset:      an offset for the geminals in g2 to indicate g2aaaa or g2bbbb blocks
        :param d1_block_id: the block id for d1a or d1b
        :param n:           the number of electrons
        """

        # g2aaaa(ij, kj) = (nmo - nalpha + 1) d1a(i, k)
        for i in range (0, self.nmo):
            for k in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for j in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    kj = self.ibas_ab[k, j]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ij + offset + 1)
                    column.append(kj + offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id)
                row.append(i + 1)
                column.append(k + 1)
                value.append(-self.nmo + n - 1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        # g2aaaa(ij, il) = nalpha djl - (nalpha - 1) d1a(l, j)
        for j in range (0, self.nmo):
            for l in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for i in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    il = self.ibas_ab[i, l]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ij + offset + 1)
                    column.append(il + offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id)
                row.append(l + 1)
                column.append(j + 1)
                value.append(n - 1.0)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(n * delta[j, l])
                self.F.append(myF)

        # g2aaaa(ij, kk) = nalpha d1a(i, j)
        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for k in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    kk = self.ibas_ab[k, k]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ij + offset + 1)
                    column.append(kk + offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id)
                row.append(i + 1)
                column.append(j + 1)
                value.append(-n)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        # g2aaaa(ii, kl) = nalpha d1a(l, k)
        for k in range (0, self.nmo):
            for l in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for i in range (0, self.nmo):
   
                    ii = self.ibas_ab[i, i]
                    kl = self.ibas_ab[k, l]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ii + offset + 1)
                    column.append(kl + offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id)
                row.append(l + 1)
                column.append(k + 1)
                value.append(-n)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def contract_g2abab_d1(self, g2_block_id, n_1, d1_block_id_1, n_2, d1_block_id_2):

        """
        contract g2ab to d1a and d1b (or g2ba to d1a and d1b) ... there are 2 different contractions

        :param g2_block_id:   the block id for g2ab or g2ba
        :param n_1:           the number of electrons for contraction 1
        :param d1_block_id_1: the block id for contraction 1
        :param n_2:           the number of electrons for contraction 1
        :param d1_block_id_2: the block id for contraction 2
        """

        # g2ab(ij, kj) = (nmo - nbeta) d1a(i, k)
        for i in range (0, self.nmo):
            for k in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for j in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    kj = self.ibas_ab[k, j]
                    block_number.append(g2_block_id)
                    row.append(ij + 1)
                    column.append(kj + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id_1)
                row.append(i + 1)
                column.append(k + 1)
                value.append(-self.nmo + n_1)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        # g2ab(ij, il) = nalpha djl - nalpha d1b(l, j)
        for j in range (0, self.nmo):
            for l in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for i in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    il = self.ibas_ab[i, l]
                    block_number.append(g2_block_id)
                    row.append(ij + 1)
                    column.append(il + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id_2)
                row.append(l + 1)
                column.append(j + 1)
                value.append(n_2)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(n_2 * delta[j, l])
                self.F.append(myF)

    def contract_g2aabb_d1(self, row_offset, col_offset, n_1, d1_block_id_1, n_2, d1_block_id_2):

        """
        contract g2aabb to d1a and d1b (or g2bbaa to d1a and d1b) ... there are 2 different contractions

        :param row_offset:    the offset for the rows of g2aaaa/aabb/bbaa/bbbb
        :param col_offset:    the offset for the columns of g2aaaa/aabb/bbaa/bbbb
        :param n_1:           the number of electrons for contraction 1
        :param d1_block_id_1: the block id for contraction 1
        :param n_2:           the number of electrons for contraction 1
        :param d1_block_id_2: the block id for contraction 2
        """

        # g2aabb(ij, kk) = nbeta d1a(i, j)
        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for k in range (0, self.nmo):
   
                    ij = self.ibas_ab[i, j]
                    kk = self.ibas_ab[k, k]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ij + row_offset + 1)
                    column.append(kk + col_offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id_1)
                row.append(i + 1)
                column.append(j + 1)
                value.append(-n_1)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

        # g2aabb(ii, kl) = nalpha d1b(l, k)
        for k in range (0, self.nmo):
            for l in range (0, self.nmo):
   
                block_number=[]
                row=[]
                column=[]
                value=[]
   
                for i in range (0, self.nmo):
   
                    ii = self.ibas_ab[i, i]
                    kl = self.ibas_ab[k, l]
                    block_number.append(self.block_id['g2aa'])
                    row.append(ii + row_offset + 1)
                    column.append(kl + col_offset + 1)
                    value.append(1.0)
   
                block_number.append(d1_block_id_2)
                row.append(l + 1)
                column.append(k + 1)
                value.append(-n_2)

                myF = libsdp.sdp_matrix()
                myF.block_number = block_number
                myF.row          = row
                myF.column       = column
                myF.value        = value
   
                self.b.append(0.0)
                self.F.append(myF)

    def trace_g2(self, offset, block_id, n):
        """
        tr(g2) = n 

        :param offset:   the offset of the row and column geminal labels
        :param block_id: the relevant block of g2 (aaaa, bbbb, abab, or baba)
        :param n:        the trace
        """
    
        block_number=[]
        row=[]
        column=[]
        value=[]
   
        for ij in range (0, len(self.bas_ab)):
            block_number.append(block_id)
            row.append(ij + offset + 1)
            column.append(ij + offset + 1)
            value.append(1.0)
   
        F = libsdp.sdp_matrix()
        F.block_number = block_number
        F.row          = row
        F.column       = column
        F.value        = value

        self.F.append(F)
        self.b.append(n)

    def g2aaaa_antisymmetry(self, offset, d1_block_id):
        """
        g2aaaa subblock should satisfy antisymmetry of d2aa

        :param offset:      the offset of the row and column geminal labels
        :param d1_block_id: the relevant block of d1 (a or b)
        """
    
        delta = np.zeros((self.nmo, self.nmo), dtype='int32')
        i = np.arange(self.nmo)
        delta[i, i] = 1

        # g2aaaa(ik, jl) + g2aaaa(ij, kl) = d1a(i, k) djl + d1a(i, j) dlk
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):

                block_number=[]
                row=[]
                column=[]
                value=[]

                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]

                ik = self.ibas_ab[i, k]
                jl = self.ibas_ab[j, l]

                block_number.append(self.block_id['g2aa'])
                row.append(ij + offset + 1)
                column.append(kl + offset + 1)
                value.append(1.0)

                block_number.append(self.block_id['g2aa'])
                row.append(ik + offset + 1)
                column.append(jl + offset + 1)
                value.append(1.0)

                block_number.append(d1_block_id)
                row.append(i + 1)
                column.append(k + 1)
                value.append(-delta[j, l])
   
                block_number.append(d1_block_id)
                row.append(i + 1)
                column.append(j + 1)
                value.append(-delta[l, k])
   
                F = libsdp.sdp_matrix()
                F.block_number = block_number
                F.row          = row
                F.column       = column
                F.value        = value

                self.F.append(F)
                self.b.append(0.0)

