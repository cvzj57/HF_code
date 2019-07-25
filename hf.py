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

    def contract_basis_function(self, list_of_primitives):
        # contract any necessary primitives.
        pass

    def vectorise_atom(self, index):
        return numpy.array([float(self.coords[index-1]['x']),
                            float(self.coords[index-1]['y']),
                            float(self.coords[index-1]['z'])])

    def measure_atom_atom_dist(self, index_1, index_2):
        return numpy.linalg.norm(self.vectorise_atom(index_2) - self.vectorise_atom(index_1))

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

    def T(self):
        # kinetic
        pass

    def E_nuc(self):
        # nuclear repulsion
        pass

    def V(self):
        # two-e bit
        pass

    def generate_new_guess(self):
        # take previous orbitals and use them to make a new guess set
        pass

    def run_SCF(self):
        this_iteration_E = 0
        last_iteration_E = 0

        def check_convergence(this_E, last_E):
            if this_E-last_E < self.convergence or self.current_iteration > self.max_iterations:
                return True
            else:
                return False

        while not check_convergence(this_iteration_E, last_iteration_E):
            pass
        # Run the SCF procedure until self-consistency is reached
        pass

if __name__ == "__main__":
    control = HartreeFocker()
    control.build_overlap_matrix()