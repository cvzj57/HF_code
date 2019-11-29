import numpy

'''Useful bits and pieces taken from Szabo and Ostlund. Especially Appendix A.

# product of gaussians A and B: g(r-R_a))g(r-R_b)=Kg(r-R_p)
# where K = exp[-αβ/(α+β)|R_a-R_b|**2]
# makes P where R_p=(aR_a+bR_b)/a+b with exp p=a+b
# => <A|B> = int{P} = exp[-αβ/(α+β)*(A−B)**2]*(π/(α+β))**3/2
# with the (2a/pi)**(3/4) normalisation constant

# Normalised gaussian 1s: phi(a,r-R_a) = (2a/pi)**(3/4)exp[-a|r-R_a|**2]

# phi_contracted(r-R_a) = sum_pL{d_p*phi_p(a_p,r-R_a)}, where L = contraction length

# For K basis functions, K**4/8 two-e integrals
'''


class BasisFunction(object):
    def __init__(self, n, coefficients, exponents, atom_hash):
        self.n = n
        self.coefficients = coefficients
        self.exponents = exponents
        self.atom_hash = atom_hash

    @staticmethod
    def norm():
        pass


class HartreeFocker:
    def __init__(self):
        self.convergence = 0.0001
        self.contracted_basis_functions = 0
        self.coords = self.read_coords()
        self.basis = self.read_basis()
        self.charge_dict = {
            'h': 1,
            'he': 2,
            'li': 3,
            'be': 4,
            'b': 5,
            'c': 6,
        }

    def read_coords(self):
        sample_coords = [
            {'x': 0.7, 'y': 0.0, 'z': 0.0, 'el': 'h', '#': 1, 'basis': 'h def-SV(P)'},
            {'x': -0.7, 'y': 0.0, 'z': 0.0, 'el': 'h', '#': 2, 'basis': 'h def-SV(P)'}
        ]
        return sample_coords

    def read_basis(self):
        # something like the basis reader I already have for MOO.
        example_basis_file_dict = {
            '$basis': {'h def-SV(P)': {
                    '3 s': [
                        {'coefficient': 0.019682158, 'exponent': 13.010701},
                        {'coefficient': 0.13796524, 'exponent': 1.9622572},
                        {'coefficient': 0.47831935, 'exponent': 0.44453796}
                    ],
                    '1 s': [
                        {'coefficient': 1.0, 'exponent': 0.12194962}
                    ]
            }
            }
        }
        return example_basis_file_dict

    def get_charge(self, atom_index):
        element = [atom['el'] for atom in self.coords if atom['#'] == atom_index][0]
        return self.charge_dict['element']

    def contract_basis_function(self, list_of_primitives):
        # contract any necessary primitives.
        pass

    def vectorise_atom(self, index):
        return numpy.array([float(self.coords[index-1]['x']),
                            float(self.coords[index-1]['y']),
                            float(self.coords[index-1]['z'])])

    def measure_atom_atom_dist(self, index_1, index_2):
        return numpy.linalg.norm(self.vectorise_atom(index_2) - self.vectorise_atom(index_1))

    def gaussian_product(self, exp_a, exp_b, index_a, index_b):
        # Product of two Gaussians is a Gaussian
        # each input is a coefficient and the coordinates of the centre

        ra = self.vectorise_atom(index_a)
        rb = self.vectorise_atom(index_b)

        exp_c = exp_a + exp_b
        diff = numpy.linalg.norm(ra - rb)**2
        N = (4*exp_a*exp_b/(numpy.pi**2))**0.75
        K = N*numpy.exp(-exp_a*exp_b/exp_c*diff)
        rc = (exp_a*ra + exp_b*rb)/exp_c

        return exp_c, diff, K, rc

    def erf(self, t):
        P = 0.3275911
        A = [0.254829592, -0.284496736, 1.42141374, 1.453152027, 1.061405429]
        T = 1.0/(1+P*t)
        Tn = T
        polynomial = A[0]*Tn
        for i in range(1,5):
            Tn = Tn*T
            polynomial = polynomial*A[i]*Tn
        return 1.0-polynomial*numpy.exp(-t*t)

    def F0(self, t):
        if t == 0:
            return 1
        else:
            return (0.5*(numpy.pi/t)**0.5)*self.erf(t**0.5)

    def calculate_normalisation_factor(self, basis_function):
        #TODO generalise to all angular momenta, this is s only.
        N = 0.0
        for primitive_i in basis_function:
            for primitive_j in basis_function:
                N_term = (primitive_i['coefficient']*primitive_j['coefficient']) / (primitive_i['exponent']+primitive_j['exponent'])**(3/2)
                N += N_term
        N = N**(-1/2)
        N *= (numpy.pi**(-3 / 4))

        # check it worked
        self_overlap = 0.0
        for primitive_i in basis_function:
            for primitive_j in basis_function:
                self_overlap += ((primitive_i['coefficient']*primitive_j['coefficient']) / (primitive_i['exponent']+primitive_j['exponent'])**(3/2))
        self_overlap *= (N**2 * numpy.pi**(3/2))

        if not 0.99999 < self_overlap < 1.00001:
            print('Normalisation problem, self overlap: ', self_overlap)

        return N

    def compute_s_overlap(self, atom_A, basis_function_A, atom_B, basis_function_B):
        S = 0.0
        ABdist = self.measure_atom_atom_dist(atom_A['#'], atom_B['#'])
        normA = self.calculate_normalisation_factor(basis_function_A)
        normB = self.calculate_normalisation_factor(basis_function_B)

        for A_primitive in basis_function_A:
            for B_primitive in basis_function_B:
                overlap = numpy.exp((-A_primitive['exponent']*B_primitive['exponent'])/(A_primitive['exponent']+B_primitive['exponent'])*ABdist**2)
                overlap *= (numpy.pi / (A_primitive['exponent'] + B_primitive['exponent'])) ** (3 / 2)
                overlap *= normA*normB*A_primitive['coefficient']*B_primitive['coefficient']
                S += overlap
        return S

    def build_overlap_matrix(self):
        # take basis functions, associate with their atoms (need distances between centres)
        basis_functions = []
        for atom in self.coords:
            basis_functions.extend([(atom, basis_function) for key, basis_function in iter(self.basis['$basis'][atom['basis']].items())])

        overlap_matrix = numpy.zeros([len(basis_functions), len(basis_functions)])
        # Iterate through them and build the overlap matrix
        for i, row_i in enumerate(basis_functions):
            for j, col_j in enumerate(basis_functions):
                overlap_matrix[i][j] = self.compute_s_overlap(row_i[0], row_i[1], col_j[0], col_j[1])
        print(overlap_matrix)
        return overlap_matrix

    def diagonalise(self, matrix):
        # diagonalise the overlap matrix? There are numpy functions for this.
        return numpy.linalg.eig(matrix)

    def new_overlap(self, exp_a, exp_b, index_a, index_b):
        exp_c, diff, K, rc = self.gaussian_product(exp_a, exp_b, index_a, index_b)
        multiplier = (numpy.pi/exp_c)**(3/2)
        return multiplier*K

    def T(self, exp_a, exp_b, index_a, index_b):
        """Kinetic energy integral"""
        exp_c, diff, K, rc = self.gaussian_product(exp_a, exp_b, index_a, index_b)
        multiplier = (numpy.pi/exp_c)**1.5

        ra = self.vectorise_atom(index_a)
        rb = self.vectorise_atom(index_b)

        reduced_exponent = exp_a*exp_b/exp_c
        return reduced_exponent*(3-2*reduced_exponent*diff)*multiplier*K

    def e_n(self, exp_a, exp_b, index_a, index_b, index_other_atom):
        """Electron-nuclear integral"""
        exp_c, diff, K, rc = self.gaussian_product(exp_a, exp_b, index_a, index_b)
        vector_other_atom = self.vectorise_atom(index_other_atom)
        Z_other_atom = self.get_charge(index_other_atom)

        return (-2*numpy.pi*Z_other_atom/exp_c)*K*self.F0(exp_c*numpy.linalg.norm(rc-vector_other_atom))

    def V(self, exp_a, exp_b, index_a, index_b, exp_c, exp_d, index_c, index_d):
        """Two-e integral"""
        exp_e, diff_ab, K_ab, re = self.gaussian_product(exp_a, exp_b, index_a, index_b)
        exp_f, diff_cd, K_cd, rf = self.gaussian_product(exp_c, exp_d, index_c, index_d)

        mutiplier = 2*numpy.pi**(5/2)*(exp_e*exp_f*(exp_e+exp_f)**(1/2))**-1
        return mutiplier*K_ab*K_cd*self.F0(exp_e*exp_f/(exp_e+exp_f)*numpy.linalg.norm(re-rf)**2)

    def run_SCF(self):

        # work out total number of basis functions for matrix size
        # take basis functions, associate with their atoms (need distances between centres)
        basis_functions = []
        for atom in self.coords:
            basis_functions.extend([(atom, basis_function) for key, basis_function in iter(self.basis['$basis'][atom['basis']].items())])

        empty_matrix = numpy.zeros([len(basis_functions), len(basis_functions)])

        # Create overlap matrix
        S = empty_matrix
        # Create kinetic matrix
        T = empty_matrix
        # Create e_n matrix
        P = empty_matrix
        # Create two-electron matrix
        V = empty_matrix

        # Iterate through them and build the matrices
        for i, row_i in enumerate(basis_functions):
            for j, col_j in enumerate(basis_functions):
                print(row_i[1])
                S[i][j] = self.new_overlap(row_i[1]['exponent'], row_i[0]['#'], col_j[1]['exponent'], col_j[0]['#'])
                T[i][j] = self.T(row_i[1]['exponent'], row_i[0]['#'], col_j[1]['exponent'], col_j[0]['#'])

        print(S)
        print(T)

        # Iterate through atoms
        # for atom in self.coords:
        #     atom_vector = self.vectorise_atom(atom['#'])
        #     atom_charge = self.get_charge(atom['#'])

        Hcore = T + P

        # Create initial Fock matrix with Hcore guess
        Fock0 = S**(-1/2)*Hcore*S**(-1/2)
        # Diagonalise
        Fock_diag = numpy.linalg.eig(Fock0)

        # Return to AO basis

        # Create density matrix

        # Compute first SCF energy

        # start iterating
        # test for convergence

        return


    def generate_new_guess(self):
        # take previous orbitals and use them to make a new guess set
        pass

    # def run_SCF(self):
    #     this_iteration_E = 0
    #     last_iteration_E = 0
    #
    #     def check_convergence(this_E, last_E):
    #         if this_E-last_E < self.convergence or self.current_iteration > self.max_iterations:
    #             return True
    #         else:
    #             return False
    #
    #     while not check_convergence(this_iteration_E, last_iteration_E):
    #         pass
    #     # Run the SCF procedure until self-consistency is reached
    #     pass

if __name__ == "__main__":
    control = HartreeFocker()
    control.run_SCF()
#    control.build_overlap_matrix()