
import numpy as np
import spectral
from scipy import sparse
import scipy.sparse.linalg as spla

class KdVEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) 
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)
        
        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        
        self.N = x_basis.N
        
        if self.dtype == np.float64:
            diag = x_basis.wavenumbers(self.dtype)**3 
            perm = np.array([[0, 1], [-1, 0]])
            perm_mat = np.kron(np.eye(self.N//2,dtype=int),perm)
            p.L = sparse.diags(diag) @ perm_mat
        
        elif self.dtype == np.complex128:
            diag = -1j * x_basis.wavenumbers(self.dtype)**3
            p.L = sparse.diags(diag)

        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate -u*ux and put it into RHS
            u.require_coeff_space()
            dudx.require_coeff_space()
            
            if self.dtype == np.float64:
                an = u.data[::2]
                bn = u.data[1::2]
                du = np.zeros(self.N, dtype=self.dtype)
                du[::2] = -bn
                du[1::2] = an
                dudx.data = x_basis.wavenumbers(self.dtype)*du #probably not correct
            elif self.dtype == np.complex128:
                dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6 * u.data * dudx.data 

            # take timestep
            ts.step(dt)

class SHEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # -u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)
        
        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N, dtype=dtype)
        p.M = I
        
        self.N = x_basis.N

        diag = 1 + 0.3 - 2 * x_basis.wavenumbers(self.dtype)**2 + x_basis.wavenumbers(self.dtype)**4 
        p.L = sparse.diags(diag)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)

        u = self.u
        RHS = self.RHS

        for i in range(num_steps):
            u.require_coeff_space()
            
            u.require_grid_space(scales=3)
            RHS.require_grid_space(scales=3)
            RHS.data = 1.8 * u.data**2 - u.data**3

            # take timestep
            ts.step(dt)


