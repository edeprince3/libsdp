"""
the semidefinite program for variational two-electron reduced density matrix theory
"""
import numpy as np

import sys
sys.path.insert(0, '../../.')

import libsdp

class v2rdm_sdp():

    def __init__(self, nalpha, nbeta, nmo, oei, tei, q2 = True, g2 = True, constrain_spin = True, t2 = False):
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
        block_id:      list of block ids (integers) for blocks of primal solution
        blocks:        list of block ids (strings) for blocks of primal solution
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
        self.dimensions.append(1) # one
        self.dimensions.append(nmo) # d1a
        self.dimensions.append(nmo) # d1b
        self.dimensions.append(nmo*nmo) # d2ab
        self.dimensions.append(nmo*(nmo-1)//2) # d2aa
        self.dimensions.append(nmo*(nmo-1)//2) # d2bb
        #self.dimensions.append(nmo) # q1a
        #self.dimensions.append(nmo) # q1b

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
        blocks = ['1', 'd1a', 'd1b', 'd2ab', 'd2aa', 'd2bb']

        if q2 :
            blocks.append('q2ab')
            blocks.append('q2aa')
            blocks.append('q2bb')

        if g2 :
            blocks.append('g2aa')
            blocks.append('g2ab')
            blocks.append('g2ba')

        # in case someone outside of the class wants to know how the blocks are ordered
        self.blocks = blocks

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
        F = libsdp.sdp_matrix()
    
        #for i in range (0, nmo):
        #    for j in range (0, nmo):
        #        block_number.append(self.block_id['d1a'])
        #        row.append(i+1)
        #        column.append(j+1)
        #        value.append(oei[i][j])
    
        #for i in range (0, nmo):
        #    for j in range (0, nmo):
        #        block_number.append(self.block_id['d1b'])
        #        row.append(i+1)
        #        column.append(j+1)
        #        value.append(oei[i][j])

        n = self.nalpha + self.nbeta
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                dum = tei[i][k][j][l]
                if i == k :
                    dum += oei[j][l] / ( n - 1 )
                if j == l :
                    dum += oei[i][k] / ( n - 1 )
                F.value.append(dum)
        
        for ij in range (0, len(self.bas_aa)):
            i = self.bas_aa[ij][0]
            j = self.bas_aa[ij][1]
            for kl in range (0, len(self.bas_aa)):
                k = self.bas_aa[kl][0]
                l = self.bas_aa[kl][1]
                F.block_number.append(self.block_id['d2aa'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                dum = tei[i][k][j][l] - tei[i][l][j][k]
                if i == k :
                    dum += oei[j][l] / ( n - 1 )
                if i == l :
                    dum -= oei[j][k] / ( n - 1 )
                if j == k :
                    dum -= oei[i][l] / ( n - 1 )
                if j == l :
                    dum += oei[i][k] / ( n - 1 )
                F.value.append(dum)
        
        for ij in range (0, len(self.bas_aa)):
            i = self.bas_aa[ij][0]
            j = self.bas_aa[ij][1]
            for kl in range (0, len(self.bas_aa)):
                k = self.bas_aa[kl][0]
                l = self.bas_aa[kl][1]
                F.block_number.append(self.block_id['d2bb'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                dum = tei[i][k][j][l] - tei[i][l][j][k]
                if i == k :
                    dum += oei[j][l] / ( n - 1 )
                if i == l :
                    dum -= oei[j][k] / ( n - 1 )
                if j == k :
                    dum -= oei[i][l] / ( n - 1 )
                if j == l :
                    dum += oei[i][k] / ( n - 1 )
                F.value.append(dum)

        self.F = []
        self.F.append(F)
        
        # build constraints (F1, F2, ...)
    
        self.b = []

        # 1 = 1
        F = libsdp.sdp_matrix()
        F.block_number.append(self.block_id['1'])
        F.row.append(1)
        F.column.append(1)
        F.value.append(1.0)

        self.F.append(F)
        self.b.append(1.0)

        # trace of d1a
        #self.trace_d1(self.block_id['d1a'], nalpha)
    
        # trace of d1b
        #self.trace_d1(self.block_id['d1b'], nbeta)

        # trace of d2ab
        self.trace_d2(self.block_id['d2ab'], self.bas_ab, nalpha*nbeta)

        # trace of d2aa
        self.trace_d2(self.block_id['d2aa'], self.bas_aa, nalpha*(nalpha-1)/2)

        # trace of d2bb
        self.trace_d2(self.block_id['d2bb'], self.bas_aa, nbeta*(nbeta-1)/2)

        # d2ab -> d1a
        self.contract_d2ab_d1a()

        # d2ab -> d1b
        self.contract_d2ab_d1b()

        # d2aa -> d1a
        self.contract_d2aa_d1a(self.block_id['d2aa'], self.block_id['d1a'], nalpha)

        # d2bb -> d1b
        self.contract_d2aa_d1a(self.block_id['d2bb'], self.block_id['d1b'], nbeta)

        # d1a <-> q1a
        #self.d1_q1_mapping(self.block_id['d1a'], self.block_id['q1a'])

        # d1b <-> q1b
        #self.d1_q1_mapping(self.block_id['d1b'], self.block_id['q1b'])

        if q2: 

            # q2ab <-> d2ab, d1a, d1b
            self.q2ab_mapping()

            # q2aa <-> d2aa, d1a
            self.q2aa_mapping(self.block_id['d2aa'], self.block_id['d1a'], self.block_id['q2aa'])

            # q2bb <-> d2bb, d1b
            self.q2aa_mapping(self.block_id['d2bb'], self.block_id['d1b'], self.block_id['q2bb'])

        if g2:

            # g2aaaa/aabb/bbaa/bbbb <-> d2ab, d2aa, d2bb, d1a, d1b
            self.g2aa_mapping()

            # g2ab <-> d2ab, d1a
            self.g2ab_mapping()

            # g2ba <-> d2ab, d1b
            self.g2ba_mapping()

        if constrain_spin :

            # <s^2> 
            self.constrain_s2()

            # spin block structure (currently only for singlets, D1a = D1b, D2aa(ijkl) = D2ab(ijkl) - D2ab(ijlk), etc.)
            #if self.nalpha == self.nbeta:
            #    self.constrain_spin_block_structure()

    def trace_d1(self, block_id, n):
        """
        tr(d1) = n

        :param block_id: the block id for this block of d1
        :param n:        the number of electrons
        """

        # Tr(D1)
        F = libsdp.sdp_matrix()
    
        for i in range (0, self.nmo):
            F.block_number.append(block_id)
            F.row.append(i+1)
            F.column.append(i+1)
            F.value.append(1.0)
    
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
        F = libsdp.sdp_matrix()
   
        F.block_number.append(self.block_id['1'])
        F.row.append(1)
        F.column.append(1)
        F.value.append(-n)

        for ij in range (0, len(bas)):
            F.block_number.append(block_id)
            F.row.append(ij+1)
            F.column.append(ij+1)
            F.value.append(1.0)
   
        self.F.append(F)
        #self.b.append(n)
        self.b.append(0.0)

    def contract_d2ab_d1a(self):
        """
        contract d2ab to d1a
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                F = libsdp.sdp_matrix()
   
                for k in range (0, self.nmo):
   
                    ik = self.ibas_ab[i, k]
                    jk = self.ibas_ab[j, k]
                    F.block_number.append(self.block_id['d2ab'])
                    F.row.append(ik+1)
                    F.column.append(jk+1)
                    F.value.append(1.0)
   
                F.block_number.append(self.block_id['d1a'])
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(-self.nbeta)

                self.b.append(0.0)
                self.F.append(F)

    def contract_d2ab_d1b(self):
        """
        contract d2ab to d1b
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                F = libsdp.sdp_matrix()
   
                for k in range (0, self.nmo):
   
                    ki = self.ibas_ab[k, i]
                    kj = self.ibas_ab[k, j]
                    F.block_number.append(self.block_id['d2ab'])
                    F.row.append(ki+1)
                    F.column.append(kj+1)
                    F.value.append(1.0)
   
                F.block_number.append(self.block_id['d1b'])
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(-self.nalpha)

                self.b.append(0.0)
                self.F.append(F)

    def contract_d2aa_d1a(self, d2_block_id, d1_block_id, n):
        """
        contract d2aa to d1a (or d2bb to d1b)

        :param d2_block_id: the block id for d2aa or d2bb
        :param d1_block_id: the block id for d1a or d1b
        :param n: the number of alpha or beta electrons
        """

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                F = libsdp.sdp_matrix()
   
                for k in range (0, self.nmo):
   
                    if i == k or j == k :
                        continue

                    ik = self.ibas_aa[i, k]
                    jk = self.ibas_aa[j, k]
                    F.block_number.append(d2_block_id)
                    F.row.append(ik+1)
                    F.column.append(jk+1)
                    sg = 1
                    if i > k :
                        sg = -sg
                    if j > k :
                        sg = -sg
                    F.value.append(sg * 1.0)
   
                F.block_number.append(d1_block_id)
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(-(n-1))

                self.b.append(0.0)
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                F.block_number.append(d1_block_id)
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(1.0)

                F.block_number.append(q1_block_id)
                F.row.append(j+1)
                F.column.append(i+1)
                F.value.append(1.0)

                self.b.append(delta[i, j])
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                # + k* l* j i
                F.block_number.append(d2_block_id)
                F.row.append(kl+1)
                F.column.append(ij+1)
                F.value.append(1.0)

                # - i j l* k*
                F.block_number.append(q2_block_id)
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(-1.0)

                # - k* i d(j,l)
                if j == l:
                    F.block_number.append(d1_block_id)
                    F.row.append(k+1)
                    F.column.append(i+1)
                    F.value.append(-1.0)

                # + k* j d(i,l)
                if i == l:
                    F.block_number.append(d1_block_id)
                    F.row.append(k+1)
                    F.column.append(j+1)
                    F.value.append(1.0)

                # + l* i d(j,k)
                if j == k:
                    F.block_number.append(d1_block_id)
                    F.row.append(l+1)
                    F.column.append(i+1)
                    F.value.append(1.0)

                # - l* j d(i,k)
                if i == k:
                    F.block_number.append(d1_block_id)
                    F.row.append(l+1)
                    F.column.append(j+1)
                    F.value.append(-1.0)

                self.b.append(- delta[i, k] * delta[j, l] + delta[i, l] * delta[j, k])
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                # + k* l* j i
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(kl+1)
                F.column.append(ij+1)
                F.value.append(1.0)

                # - i j l* k*
                F.block_number.append(self.block_id['q2ab'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(-1.0)

                # - k* i d(j,l)
                if j == l:
                    F.block_number.append(self.block_id['d1a'])
                    F.row.append(k+1)
                    F.column.append(i+1)
                    F.value.append(-1.0)

                # - l* j d(i,k)
                if i == k:
                    F.block_number.append(self.block_id['d1b'])
                    F.row.append(l+1)
                    F.column.append(j+1)
                    F.value.append(-1.0)

                # - d(j,l) d(i,k) 
                self.b.append(- delta[i, k] * delta[j, l])
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                # - i* l* j k
                il = self.ibas_ab[i, l]
                kj = self.ibas_ab[k, j]
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(il+1)
                F.column.append(kj+1)
                F.value.append(-1.0)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2ab'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    F.block_number.append(self.block_id['d1a'])
                    F.row.append(i+1)
                    F.column.append(k+1)
                    F.value.append(1.0)

                self.b.append(0.0)
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                # - l* i* k j
                li = self.ibas_ab[l, i]
                jk = self.ibas_ab[j, k]
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(li+1)
                F.column.append(jk+1)
                F.value.append(-1.0)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2ba'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    F.block_number.append(self.block_id['d1b'])
                    F.row.append(i+1)
                    F.column.append(k+1)
                    F.value.append(1.0)

                self.b.append(0.0)
                self.F.append(F)

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

        # aaaa:
        # 0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                F = libsdp.sdp_matrix()
   
                # - i* l* j k
                if i != l and k != j:
                    il = self.ibas_aa[i][l]
                    kj = self.ibas_aa[k][j]
                    F.block_number.append(self.block_id['d2aa'])
                    F.row.append(il+1)
                    F.column.append(kj+1)
                    sg = 1
                    if i > l:
                        sg = -sg
                    if k > j:
                        sg = -sg
                    F.value.append(-sg)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2aa'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    F.block_number.append(self.block_id['d1a'])
                    F.row.append(i+1)
                    F.column.append(k+1)
                    F.value.append(1.0)

                self.b.append(0.0)
                self.F.append(F)

        # bbbb:
        # 0 = - i* l* j k  - i* j l* k  + i* k d(j,l) 
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                F = libsdp.sdp_matrix()
   
                # - i* l* j k
                if i != l and k != j:
                    il = self.ibas_aa[i][l]
                    kj = self.ibas_aa[k][j]
                    F.block_number.append(self.block_id['d2bb'])
                    F.row.append(il+1)
                    F.column.append(kj+1)
                    sg = 1
                    if i > l:
                        sg = -sg
                    if k > j:
                        sg = -sg
                    F.value.append(-sg)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2aa'])
                F.row.append(len(self.bas_ab) + ij + 1)
                F.column.append(len(self.bas_ab) + kl + 1)
                F.value.append(-1.0)

                # + i* k d(j,l)
                if j == l:
                    F.block_number.append(self.block_id['d1b'])
                    F.row.append(i+1)
                    F.column.append(k+1)
                    F.value.append(1.0)

                self.b.append(0.0)
                self.F.append(F)

        # aabb:
        # 0 = + i* l* k j  - i* j l* k
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                F = libsdp.sdp_matrix()
   
                # + i* l* k j
                il = self.ibas_ab[i, l]
                jk = self.ibas_ab[j, k]
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(il+1)
                F.column.append(jk+1)
                F.value.append(1.0)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2aa'])
                F.row.append(ij + 1)
                F.column.append(len(self.bas_ab) + kl + 1)
                F.value.append(-1.0)

                self.b.append(0.0)
                self.F.append(F)

        # bbaa:
        # 0 = + l* i* j k  - i* j l* k
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
   
                F = libsdp.sdp_matrix()
   
                # + l* i* j k
                li = self.ibas_ab[l, i]
                kj = self.ibas_ab[k, j]
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(li+1)
                F.column.append(kj+1)
                F.value.append(1.0)

                # - i* j l* k 
                F.block_number.append(self.block_id['g2aa'])
                F.row.append(len(self.bas_ab) + ij + 1)
                F.column.append(kl + 1)
                F.value.append(-1.0)

                self.b.append(0.0)
                self.F.append(F)

    def constrain_s2(self):
        """
        constrain <s^2>
        """

        ms = 0.5 * (self.nalpha - self.nbeta)

        F = libsdp.sdp_matrix()

        for i in range (0, self.nmo):
            for j in range (0, self.nmo):
   
                ij = self.ibas_ab[i, j]
                ji = self.ibas_ab[j, i]

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij+1)
                F.column.append(ji+1)
                F.value.append(1.0)

        self.b.append(0.5 * (self.nalpha + self.nbeta) + ms*ms - ms*(ms+1.0))
        self.F.append(F)

        # maximal spin 
        #for i in range (0, self.nmo):
        #    for j in range (0, self.nmo):
        #        ij = self.ibas_ab[i, j]

        #        F = libsdp.sdp_matrix()

        #        for k in range (0, self.nmo):
        #            kk = self.ibas_ab[k, k]
        #            F.block_number.append(self.block_id['g2ba'])
        #            F.row.append(kk+1)
        #            F.column.append(ij+1)
        #            F.value.append(1.0)

        #        self.b.append(0.0)
        #        self.F.append(F)

        ## maximal spin 
        #for i in range (0, self.nmo):
        #    for j in range (0, self.nmo):
        #        ij = self.ibas_ab[i, j]

        #        F = libsdp.sdp_matrix()

        #        for k in range (0, self.nmo):
        #            kk = self.ibas_ab[k, k]

        #            F.block_number.append(self.block_id['g2ba'])
        #            F.row.append(ij+1)
        #            F.column.append(kk+1)
        #            F.value.append(1.0)

        #        self.b.append(0.0)
        #        self.F.append(F)

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
   
                F = libsdp.sdp_matrix()

                F.block_number.append(self.block_id['d1a'])
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(1.0)

                F.block_number.append(self.block_id['d1b'])
                F.row.append(i+1)
                F.column.append(j+1)
                F.value.append(-1.0)

                self.b.append(0.0)
                self.F.append(F)

        # d2ab = d2ba
        for ij in range (0, len(self.bas_ab)):
            i = self.bas_ab[ij][0]
            j = self.bas_ab[ij][1]
            ji = self.ibas_ab[j, i]
            for kl in range (0, len(self.bas_ab)):
                k = self.bas_ab[kl][0]
                l = self.bas_ab[kl][1]
                lk = self.ibas_ab[l, k]
   
                F = libsdp.sdp_matrix()
   
                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(1.0)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ji+1)
                F.column.append(lk+1)
                F.value.append(-1.0)

                self.b.append(0.0)
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                F.block_number.append(self.block_id['d2aa'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(1.0)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij_ab+1)
                F.column.append(kl_ab+1)
                F.value.append(-0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij_ab+1)
                F.column.append(lk_ab+1)
                F.value.append(0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ji_ab+1)
                F.column.append(kl_ab+1)
                F.value.append(0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ji_ab+1)
                F.column.append(lk_ab+1)
                F.value.append(-0.5)

                self.b.append(0.0)
                self.F.append(F)

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
   
                F = libsdp.sdp_matrix()
   
                F.block_number.append(self.block_id['d2bb'])
                F.row.append(ij+1)
                F.column.append(kl+1)
                F.value.append(1.0)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij_ab+1)
                F.column.append(kl_ab+1)
                F.value.append(-0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ij_ab+1)
                F.column.append(lk_ab+1)
                F.value.append(0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ji_ab+1)
                F.column.append(kl_ab+1)
                F.value.append(0.5)

                F.block_number.append(self.block_id['d2ab'])
                F.row.append(ji_ab+1)
                F.column.append(lk_ab+1)
                F.value.append(-0.5)

                self.b.append(0.0)
                self.F.append(F)

    def get_rdm_blocks(self, x):
        """
        extract rdm/integral blocks from a vector of dimension of primal solution

        :param x: the vector or rdm/integral elements
        """

        rdms = []

        off = 0
        for i in range (0, len(self.dimensions)):
            rdms.append(np.array(x[off:off + self.dimensions[i]**2]).reshape(self.dimensions[i], self.dimensions[i]))
            off = off + self.dimensions[i]**2

        return rdms

    def get_constraint_matrix(self, constraint_id):
        """
        return a row of the constraint matrix, in non-sparse format

        :param constraint_id: the row of the constraint matrix to be returned (zero offset)
        """

        n_primal = 0
        for i in range (0, len(self.dimensions)):
            n_primal = n_primal + self.dimensions[i]**2

        a = np.zeros(n_primal, dtype = 'float64')

        for i in range (0, self.F[constraint_id + 1].block_number.size()):
            my_block = self.F[constraint_id + 1].block_number[i] - 1
            my_row = self.F[constraint_id + 1].row[i] - 1
            my_col = self.F[constraint_id + 1].column[i] - 1

            # calculate offset
            off = 0
            for j in range (0, my_block):
                off = off + self.dimensions[j]**2

            my_id = off + my_row * self.dimensions[my_block] + my_col;

            a[my_id] = self.F[constraint_id + 1].value[i]

        return a
