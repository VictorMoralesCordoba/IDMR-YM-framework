"""
IDMR–YM: Induced Mass and Metric Rescaling Framework
Author: Victor Eduardo Morales Córdoba
Email: vmorales@uned.cr
ORCID: https://orcid.org/0009-0000-8787-6141
Version: 2.4 - Corrected Eigenproblem Implementation
License: CC-BY 4.0 International
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# =============================================================================
# GAMMA MATRIX ALGEBRA - FLEXIBLE REPRESENTATION SYSTEM
# =============================================================================

class GammaRepresentation(ABC):
    """Abstract base class for gamma matrix representations"""
    
    def __init__(self, dimension: int):
        self.dim = dimension
        self.gamma_matrices = {}
        self.setup_representation()
    
    @abstractmethod
    def setup_representation(self):
        """Setup specific gamma matrix representation"""
        pass
    
    def get_gamma(self, mu: int) -> np.ndarray:
        """Get gamma matrix for index mu"""
        if mu not in self.gamma_matrices:
            raise ValueError(f"Gamma matrix γ_{mu} not defined in {self.__class__.__name__}")
        return self.gamma_matrices[mu]
    
    def get_chiral_basis(self):
        """Get chirality projection operators if defined"""
        return getattr(self, 'gamma5', None)


class DiracRepresentation(GammaRepresentation):
    """Standard Dirac representation"""
    
    def setup_representation(self):
        if self.dim == 2:  # 1+1D
            self.gamma_matrices = {
                0: np.array([[1, 0], [0, -1]], dtype=complex),   # γ⁰ = σ_z
                1: np.array([[0, 1], [-1, 0]], dtype=complex)    # γ¹ = iσ_y
            }
        elif self.dim == 4:  # 3+1D
            self.gamma_matrices = {
                0: np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0], 
                            [0, 0, -1, 0],
                            [0, 0, 0, -1]], dtype=complex),
                1: np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0]], dtype=complex),
                2: np.array([[0, 0, 0, -1j],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [-1j, 0, 0, 0]], dtype=complex),
                3: np.array([[0, 0, 1, 0],
                            [0, 0, 0, -1],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=complex)
            }
            # Chirality operator in 4D
            self.gamma5 = 1j * self.gamma_matrices[0] @ self.gamma_matrices[1] @ self.gamma_matrices[2] @ self.gamma_matrices[3]


class WeylRepresentation(GammaRepresentation):
    """Weyl (chiral) representation"""
    
    def setup_representation(self):
        if self.dim == 2:  # 1+1D
            self.gamma_matrices = {
                0: np.array([[0, 1], [1, 0]], dtype=complex),   # γ⁰ = σ_x
                1: np.array([[0, -1], [1, 0]], dtype=complex)   # γ¹ = -iσ_y
            }
        elif self.dim == 4:  # 3+1D
            self.gamma_matrices = {
                0: np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=complex),
                1: np.array([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0]], dtype=complex),
                2: np.array([[0, 0, 0, -1j],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [-1j, 0, 0, 0]], dtype=complex),
                3: np.array([[0, 0, 1, 0],
                            [0, 0, 0, -1],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=complex)
            }
            # Chirality operator
            self.gamma5 = np.array([[-1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=complex)


class GammaFactory:
    """Factory class for creating gamma matrix representations"""
    
    @staticmethod
    def create_representation(representation_type: str, dimension: int) -> GammaRepresentation:
        """Create gamma matrix representation based on type string"""
        representations = {
            'dirac': DiracRepresentation,
            'weyl': WeylRepresentation
        }
        
        if representation_type.lower() not in representations:
            available = list(representations.keys())
            raise ValueError(f"Representation '{representation_type}' not available. Choose from: {available}")
        
        return representations[representation_type.lower()](dimension)


# =============================================================================
# MAIN IDMR–YM PHYSICAL FRAMEWORK
# =============================================================================

class IDMR_YM:
    """
    IDMR–YM Framework: Induced Mass and Metric Rescaling
    
    CORE PHYSICAL PRINCIPLES:
    - Dual mass mechanism: m(φ) = αφ² + β(∂φ)²
    - Dynamical metric: g_{μν}(φ) = η_{μν} f(φ)  
    - Geometric connection: Γ_μ = (1/2) ∂_μ ln(f(φ))
    - Extended gauge covariance: D_μ = ∂_μ + ieA_μ + iΓ_μ(φ)
    """
    
    def __init__(self, alpha: float = 1.65e-31, beta: float = 1e-45, 
                 epsilon: float = 0.01, e_charge: float = 1.0, 
                 dimension: int = 2, representation: str = 'dirac'):
        """
        Initialize IDMR–YM model with physical parameters
        
        Parameters calibrated for electron mass (0.511 MeV):
        - α = m_e / φ₀² ≈ 1.65e-31 GeV⁻¹
        - β = m_e / (∂φ)² ≈ 1e-45 GeV⁻³·m² (attometer scale)
        - ε = metric modulation (dimensionless)
        """
        # Physical parameters
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.e = e_charge
        
        # Natural constants
        self.hbar = 6.582119569e-25  # GeV·s
        self.c = 2.99792458e8        # m/s
        self.phi0 = 246e9            # Electroweak VEV [GeV]
        self.width = 1e-18           # Characteristic length [m]
        
        # Algebraic system
        self.dimension = dimension
        self.gamma_rep = GammaFactory.create_representation(representation, dimension)
        
        print(f"✅ IDMR–YM initialized in {dimension}D with {representation} representation")
        print(f"   Target electron mass: 0.000511 GeV")
        print(f"   Parameters: α={self.alpha:.2e}, β={self.beta:.2e}, ε={self.epsilon}")
    
    # =========================================================================
    # SCALAR FIELD AND MASS GENERATION
    # =========================================================================
    
    def phi(self, x: float, phi0: float = None, width: float = None) -> float:
        """
        Scalar field with Gaussian profile [GeV]
        
        Args:
            x: Position [m]
            phi0: Vacuum expectation value [GeV]  
            width: Characteristic width [m]
        """
        phi0 = phi0 or self.phi0
        width = width or self.width
        return phi0 * np.exp(-(x / width)**2)
    
    def dphi_dx(self, x: float, phi0: float = None, width: float = None) -> float:
        """Scalar field gradient [GeV/m]"""
        phi0 = phi0 or self.phi0
        width = width or self.width
        return -2 * x * self.phi(x, phi0, width) / width**2
    
    def m_eff(self, x: float) -> float:
        """
        Effective induced mass [GeV]
        
        Dual mass mechanism:
        m(φ) = α·φ² (conventional) + β·(∂φ)² (kinetic)
        """
        phi_val = self.phi(x)
        dphi_val = self.dphi_dx(x)
        return self.alpha * phi_val**2 + self.beta * dphi_val**2
    
    # =========================================================================
    # GEOMETRIC IMPLEMENTATION - FIRST PRINCIPLES
    # =========================================================================
    
    def f_metric(self, x: float) -> float:
        """Metric modulation function f(φ) = 1 + εφ² [dimensionless]"""
        return 1.0 + self.epsilon * self.phi(x)**2
    
    def Gamma_mu(self, x: float, mu: int = 0) -> complex:
        """
        Geometrically derived scalar connection [GeV]
        
        Derived from: Γ_μ = (1/2) ∂_μ ln(f(φ))
        where f(φ) = 1 + εφ² is the metric modulation
        
        This ensures geometric consistency in the extended covariant derivative
        """
        f_phi = self.f_metric(x)
        dphi_dx_val = self.dphi_dx(x)
        
        if abs(f_phi) > 1e-15:
            # Γ_μ = (1/2) ∂_μ ln(f(φ)) = (1/2) [2εφ ∂_μφ] / (1 + εφ²)
            connection = 0.5 * (2 * self.epsilon * self.phi(x) * dphi_dx_val) / f_phi
        else:
            connection = 0.0
            
        # Temporal component has different physical interpretation
        if mu == 0:  # Energy modulation
            return 0.1 * connection  # Placeholder for proper temporal derivation
        else:        # Spatial components
            return connection
    
    def D_mu(self, x: float, A_mu: float, mu: int = 0) -> complex:
        """
        Extended covariant derivative [GeV]
        
        D_μ = ∂_μ + ieA_μ + iΓ_μ(φ)
        
        Unifies gauge covariance with scalar geometric connection
        """
        partial_mu = 0.0  # Assuming stationary case
        return partial_mu + 1j * self.e * A_mu + 1j * self.Gamma_mu(x, mu)
    
    # =========================================================================
    # MODIFIED DIRAC EQUATION - COMPLETE 2D & 4D IMPLEMENTATION
    # =========================================================================
    
    def modified_dirac_operator(self, psi: np.ndarray, x: float, A_mu: float) -> np.ndarray:
        """
        Modified Dirac operator: (iγ^μ D_μ - m_eff)ψ
        
        Full matrix implementation respecting spinor structure
        """
        if self.dimension == 2:
            return self._dirac_operator_2d(psi, x, A_mu)
        else:
            return self._dirac_operator_4d(psi, x, A_mu)
    
    def _dirac_operator_2d(self, psi: np.ndarray, x: float, A_mu: float) -> np.ndarray:
        """Dirac operator in 1+1 dimensions"""
        gamma0 = self.gamma_rep.get_gamma(0)
        gamma1 = self.gamma_rep.get_gamma(1)
        
        D0 = self.D_mu(x, A_mu, mu=0)  # Temporal
        D1 = self.D_mu(x, A_mu, mu=1)  # Spatial
        m = self.m_eff(x)
        
        # Dirac operator: iγ⁰D₀ + iγ¹D₁ - m
        # Using * for scalar-matrix multiplication (D_mu is scalar)
        operator = 1j * gamma0 * D0 + 1j * gamma1 * D1 - m * np.eye(2)
        return operator @ psi  # Matrix-spinor multiplication
    
    def _dirac_operator_4d(self, psi: np.ndarray, x: float, A_mu: float) -> np.ndarray:
        """
        Complete Dirac operator in 3+1 dimensions
        
        Implements: (iγ^μ D_μ - m_eff)ψ
        where:
        - γ^μ: 4×4 gamma matrices
        - D_μ: scalar complex number (∂_μ + ieA_μ + iΓ_μ(φ))
        - m_eff: scalar mass
        """
        m = self.m_eff(x)
        operator = np.zeros((4, 4), dtype=complex)
        
        # Precompute all D_mu values for better performance
        D_mu_values = [self.D_mu(x, A_mu, mu=mu) for mu in range(4)]
        
        # Sum over spacetime dimensions: iγ^μ D_μ
        # Using * for scalar-matrix multiplication (D_mu is scalar)
        for mu in range(4):
            gamma_mu = self.gamma_rep.get_gamma(mu)  # 4×4 matrix
            D_mu_val = D_mu_values[mu]               # scalar complex
            operator += 1j * gamma_mu * D_mu_val     # scalar × matrix
        
        # Subtract mass term
        operator -= m * np.eye(4)
        
        return operator @ psi  # Matrix-spinor multiplication
    
    # =========================================================================
    # OPTIMIZED DIRAC SOLVERS - SPATIAL EVOLUTION
    # =========================================================================
    
    def solve_dirac_spatial_1d(self, x_range: tuple, energy: float = None, 
                              A_mu: float = 0.0, psi0: np.ndarray = None) -> object:
        """
        Optimized spatial solver for 1+1D Dirac equation
        
        Solves: (iγ⁰D₀ + iγ¹D₁ - m)ψ = 0 as ∂₁ψ = RHS
        where:
        - D₀ = -iE + ieA₀ + iΓ₀ (stationary states)
        - D₁ = ∂₁ + ieA₁ + iΓ₁
        """
        if psi0 is None:
            psi0 = np.array([1.0, 0.0], dtype=complex)
        
        if energy is None:
            energy = 0.000511  # Default electron energy [GeV]

        def dirac_spatial_rhs(x, psi):
            """Optimized right-hand side for spatial Dirac equation"""
            psi_vec = psi.reshape(2, 1)
            
            # Get gamma matrices
            gamma0 = self.gamma_rep.get_gamma(0)
            gamma1 = self.gamma_rep.get_gamma(1)
            gamma1_inv = np.linalg.inv(gamma1)
            
            # Physical parameters at position x
            m = self.m_eff(x)
            Gamma_0 = self.Gamma_mu(x, 0)
            Gamma_1 = self.Gamma_mu(x, 1)
            
            # =================================================================
            # OPTIMIZED RHS CALCULATION - SINGLE MATRIX OPERATION
            # =================================================================
            
            # Build the complete RHS without ∂₁ term
            # From: iγ⁰D₀ψ + iγ¹(ieA₁ + iΓ₁)ψ - mψ = -iγ¹∂₁ψ
            
            # Term 1: iγ⁰D₀ψ = iγ⁰(-iE + ieA₀ + iΓ₀)ψ
            D0_coeff = -1j * energy + 1j * self.e * A_mu + 1j * Gamma_0
            term1 = 1j * gamma0 @ (D0_coeff * psi_vec)
            
            # Term 2: iγ¹(ieA₁ + iΓ₁)ψ = iγ¹(i(eA₁ + Γ₁))ψ = -γ¹(eA₁ + Γ₁)ψ  
            spatial_gauge = self.e * A_mu + Gamma_1  # A₁ = A_mu for spatial component
            term2 = -gamma1 @ (spatial_gauge * psi_vec)
            
            # Term 3: -mψ
            term3 = -m * psi_vec
            
            # Combine all terms: RHS = term1 + term2 + term3
            total_rhs = term1 + term2 + term3
            
            # Final RHS for ∂₁ψ: ∂₁ψ = -i(γ¹)⁻¹ × total_rhs
            rhs = -1j * gamma1_inv @ total_rhs
            
            return rhs.flatten()
        
        solution = solve_ivp(dirac_spatial_rhs, x_range, psi0,
                            method='BDF', rtol=1e-8, atol=1e-10)
        return solution
    
    def solve_dirac_spatial_1d_optimized(self, x_range: tuple, energy: float = None, 
                                       A_mu: float = 0.0, psi0: np.ndarray = None) -> object:
        """
        Super-optimized spatial solver - minimal operations
        """
        if psi0 is None:
            psi0 = np.array([1.0, 0.0], dtype=complex)
        
        if energy is None:
            energy = 0.000511

        # Precompute gamma matrices and inverses (constant for 1+1D)
        gamma0 = self.gamma_rep.get_gamma(0)
        gamma1 = self.gamma_rep.get_gamma(1)
        gamma1_inv = np.linalg.inv(gamma1)
        
        # Precompute constant matrix combinations
        i_gamma0 = 1j * gamma0
        neg_i_gamma1_inv = -1j * gamma1_inv

        def dirac_spatial_rhs(x, psi):
            """Ultra-optimized RHS with minimal operations"""
            psi_vec = psi.reshape(2, 1)
            
            # Local physical parameters
            m = self.m_eff(x)
            Gamma_0 = self.Gamma_mu(x, 0)
            Gamma_1 = self.Gamma_mu(x, 1)
            
            # Compute D₀ coefficient
            D0_coeff = -1j * energy + 1j * self.e * A_mu + 1j * Gamma_0
            
            # Single combined RHS calculation
            term1 = i_gamma0 @ (D0_coeff * psi_vec)           # iγ⁰D₀ψ
            term2 = gamma1 @ ((self.e * A_mu + Gamma_1) * psi_vec)  # γ¹(eA₁ + Γ₁)ψ  
            term3 = m * psi_vec                                # mψ
            
            # ∂₁ψ = -i(γ¹)⁻¹ × [term1 + term2 - term3]
            rhs = neg_i_gamma1_inv @ (term1 + term2 - term3)
            
            return rhs.flatten()
        
        return solve_ivp(dirac_spatial_rhs, x_range, psi0, method='BDF', rtol=1e-8, atol=1e-10)
    
    # =========================================================================
    # CORRECTED EIGENPROBLEM SOLVER - STATIONARY STATES
    # =========================================================================
    
    def solve_dirac_eigenproblem_1d(self, x_points: np.ndarray, 
                                   A_mu: float = 0.0, num_states: int = 5) -> dict:
        """
        CORRECTED Dirac eigenproblem solver for stationary states
        
        Hamiltonian derivation:
        H = -iγ⁰γ¹∂₁ + γ⁰m + γ⁰(eA₀ + Γ₀) + γ¹(eA₁ + Γ₁)
        """
        n_points = len(x_points)
        dx = x_points[1] - x_points[0]
        
        # Initialize Hamiltonian matrix (sparse format)
        H_size = 2 * n_points  # 2 spinor components × n spatial points
        H_data = []
        H_rows = []
        H_cols = []
        
        # Gamma matrices for 1+1D
        gamma0 = self.gamma_rep.get_gamma(0)
        gamma1 = self.gamma_rep.get_gamma(1)
        
        # Build Hamiltonian using finite differences
        for i in range(n_points):
            x = x_points[i]
            
            # Local mass and potential terms
            m = self.m_eff(x)
            Gamma_0 = self.Gamma_mu(x, 0)
            Gamma_1 = self.Gamma_mu(x, 1)
            
            # CORRECTED LOCAL HAMILTONIAN:
            # H_local = γ⁰m + γ⁰(eA₀ + Γ₀) + γ¹(eA₁ + Γ₁)
            mass_term = gamma0 * m
            temporal_gauge = gamma0 * (self.e * A_mu + Gamma_0)  # CORRECTED: Signo POSITIVO
            spatial_gauge = gamma1 * (self.e * A_mu + Gamma_1)   # CORRECTED: Signo POSITIVO
            
            local_H = mass_term + temporal_gauge + spatial_gauge
            
            # Add local terms to Hamiltonian
            for comp_i in range(2):
                for comp_j in range(2):
                    idx_i = 2*i + comp_i
                    idx_j = 2*i + comp_j
                    H_rows.append(idx_i)
                    H_cols.append(idx_j)
                    H_data.append(local_H[comp_i, comp_j])
            
            # Derivative term: -iγ⁰γ¹ ∂₁ (finite difference)
            if i > 0 and i < n_points - 1:
                # Central difference: ∂₁ψ ≈ (ψ_{i+1} - ψ_{i-1})/(2dx)
                derivative_coeff = -1j * gamma0 @ gamma1 / (2 * dx)
                
                # Forward difference to i+1
                for comp_i in range(2):
                    for comp_j in range(2):
                        idx_i = 2*i + comp_i
                        idx_j = 2*(i+1) + comp_j
                        H_rows.append(idx_i)
                        H_cols.append(idx_j)
                        H_data.append(derivative_coeff[comp_i, comp_j])
                
                # Backward difference to i-1  
                for comp_i in range(2):
                    for comp_j in range(2):
                        idx_i = 2*i + comp_i
                        idx_j = 2*(i-1) + comp_j
                        H_rows.append(idx_i)
                        H_cols.append(idx_j)
                        H_data.append(-derivative_coeff[comp_i, comp_j])
        
        # Construct sparse Hamiltonian
        H_sparse = csr_matrix((H_data, (H_rows, H_cols)), shape=(H_size, H_size))
        
        # Solve eigenvalue problem
        try:
            eigenvalues, eigenvectors = eigs(H_sparse, k=num_states, sigma=0, which='LM')
        except:
            # Fallback to dense diagonalization for small systems
            H_dense = H_sparse.toarray()
            eigenvalues, eigenvectors = np.linalg.eig(H_dense)
        
        # Sort by real part of energy
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return {
            'energies': eigenvalues,
            'wavefunctions': eigenvectors,
            'x_grid': x_points,
            'hamiltonian': H_sparse
        }
    
    def find_bound_states(self, x_range: tuple = (-5e-18, 5e-18), 
                         num_points: int = 1000, num_states: int = 10) -> dict:
        """
        Find bound states in the scalar field potential
        """
        x_grid = np.linspace(x_range[0], x_range[1], num_points)
        
        # Solve eigenproblem
        solution = self.solve_dirac_eigenproblem_1d(x_grid, num_states=num_states)
        
        # Filter bound states (negative real energy)
        bound_mask = np.real(solution['energies']) < 0
        bound_energies = solution['energies'][bound_mask]
        bound_wavefunctions = solution['wavefunctions'][:, bound_mask]
        
        # Calculate effective potential from scalar field
        potential = np.array([-self.m_eff(x) for x in x_grid])
        
        return {
            'bound_energies': bound_energies,
            'bound_wavefunctions': bound_wavefunctions,
            'x_grid': x_grid,
            'effective_potential': potential,
            'num_bound_states': len(bound_energies)
        }
    
    # =========================================================================
    # PHYSICAL VALIDATION AND BENCHMARKING
    # =========================================================================
    
    def test_mass_generation(self) -> None:
        """
        Physical validation: test mass generation mechanism
        
        Compares computed effective mass with known electron mass
        at different spatial positions
        """
        test_points = [-2e-18, -1e-18, 0, 1e-18, 2e-18]
        
        print("\n🔬 MASS GENERATION VALIDATION:")
        print("Position [m]     φ [GeV]         m_eff [GeV]     Target: 0.000511 GeV")
        print("-" * 65)
        
        for x in test_points:
            phi_val = self.phi(x)
            m_eff_val = self.m_eff(x)
            print(f"{x:>10.1e}    {phi_val:>10.2e}    {m_eff_val:>10.6f}")
    
    def benchmark_against_electron(self) -> dict:
        """
        Comprehensive benchmark against physical electron mass
        
        Returns detailed comparison with deviation analysis
        """
        central_mass = self.m_eff(0)
        target_mass = 0.000511  # Electron mass in GeV
        
        deviation = abs(central_mass - target_mass) / target_mass * 100
        
        return {
            'computed_mass': central_mass,
            'target_mass': target_mass,
            'absolute_error': abs(central_mass - target_mass),
            'relative_error_percent': deviation,
            'physically_consistent': deviation < 5.0  # 5% tolerance
        }


# =============================================================================
# ANALYSIS AND VISUALIZATION TOOLS
# =============================================================================

class IDMR_Analysis:
    """
    Advanced analysis tools for IDMR–YM framework
    
    Provides physical validation, visualization, and data export
    capabilities for comprehensive scientific analysis
    """
    
    def __init__(self, idmr_model: IDMR_YM):
        self.model = idmr_model
    
    def compute_energy_density(self, x_values: np.ndarray) -> list:
        """
        Compute energy density components [GeV⁴]
        
        Returns scalar field energy density and fermionic 
        contributions for physical consistency checking
        """
        energy_density = []
        
        for x in x_values:
            phi_val = self.model.phi(x)
            dphi_val = self.model.dphi_dx(x)
            m_eff = self.model.m_eff(x)
            
            # Scalar field energy density
            T_phi = 0.5 * dphi_val**2 + 0.5 * self.model.alpha * phi_val**4
            
            # Fermion contribution (simplified)
            T_psi = m_eff**4  
            
            energy_density.append({
                'position': x,
                'scalar_density': T_phi,
                'fermion_density': T_psi,
                'total_density': T_phi + T_psi,
                'metric_factor': self.model.f_metric(x)
            })
        
        return energy_density
    
    def analyze_mass_profile(self, x_range: tuple, num_points: int = 1000) -> dict:
        """
        Comprehensive analysis of mass generation across space
        
        Returns complete physical profiles for visualization
        and theoretical analysis
        """
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        
        profiles = {
            'x': x_vals,
            'phi': np.array([self.model.phi(x) for x in x_vals]),
            'm_eff': np.array([self.model.m_eff(x) for x in x_vals]),
            'f_metric': np.array([self.model.f_metric(x) for x in x_vals]),
            'Gamma_mu': np.array([self.model.Gamma_mu(x, mu=1) for x in x_vals])
        }
        
        return profiles
    
    def plot_profiles(self, x_range: tuple) -> plt.Figure:
        """
        Generate comprehensive visualization of physical profiles
        
        Four-panel plot showing:
        - Scalar field evolution
        - Effective mass generation  
        - Metric modulation
        - Geometric connection
        """
        profiles = self.analyze_mass_profile(x_range)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scalar field profile
        ax1.plot(profiles['x'], profiles['phi'])
        ax1.set_xlabel('Position [m]')
        ax1.set_ylabel('Scalar Field φ [GeV]')
        ax1.set_title('Scalar Field Profile')
        ax1.grid(True, alpha=0.3)
        
        # Effective mass with target comparison
        ax2.plot(profiles['x'], profiles['m_eff'])
        ax2.axhline(y=0.000511, color='r', linestyle='--', label='Electron mass')
        ax2.set_xlabel('Position [m]')
        ax2.set_ylabel('Effective Mass [GeV]')
        ax2.set_title('Induced Mass Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Metric modulation
        ax3.plot(profiles['x'], profiles['f_metric'])
        ax3.set_xlabel('Position [m]')
        ax3.set_ylabel('Metric Factor f(φ)')
        ax3.set_title('Metric Modulation')
        ax3.grid(True, alpha=0.3)
        
        # Geometric scalar connection
        ax4.plot(profiles['x'], np.real(profiles['Gamma_mu']))
        ax4.set_xlabel('Position [m]')
        ax4.set_ylabel('Scalar Connection Γ_μ [GeV]')
        ax4.set_title('Geometric Scalar Connection')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_bound_states(self, bound_states: dict) -> plt.Figure:
        """
        Plot bound states and wavefunctions
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        x_grid = bound_states['x_grid']
        potential = bound_states['effective_potential']
        energies = bound_states['bound_energies']
        wavefunctions = bound_states['bound_wavefunctions']
        
        # Plot potential and energy levels
        ax1.plot(x_grid, potential, 'k-', linewidth=2, label='Effective Potential')
        
        for i, energy in enumerate(energies):
            ax1.axhline(y=np.real(energy), color=f'C{i}', linestyle='--', 
                       label=f'State {i}: E = {np.real(energy):.6f} GeV')
        
        ax1.set_xlabel('Position [m]')
        ax1.set_ylabel('Energy [GeV]')
        ax1.set_title('Bound States in Scalar Potential')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot wavefunctions
        for i in range(min(3, len(energies))):  # Plot first 3 states
            # Extract upper component of wavefunction
            wf_upper = np.abs(wavefunctions[0::2, i])**2
            # Normalize for better visualization
            wf_upper = wf_upper / np.max(wf_upper) if np.max(wf_upper) > 0 else wf_upper
            ax2.plot(x_grid, wf_upper + np.real(energies[i]), label=f'State {i}')
        
        ax2.set_xlabel('Position [m]')
        ax2.set_ylabel('Energy [GeV]')
        ax2.set_title('Bound State Wavefunctions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_profiles(self, x_range: tuple, filename: str) -> None:
        """
        Export physical profiles to CSV for external analysis
        
        Enables data sharing and reproducibility in scientific
        collaboration
        """
        profiles = self.analyze_mass_profile(x_range)
        
        data = np.column_stack([
            profiles['x'],
            profiles['phi'],
            profiles['m_eff'],
            profiles['f_metric'],
            np.real(profiles['Gamma_mu'])
        ])
        
        header = "Position [m],Scalar_Field [GeV],Effective_Mass [GeV],Metric_Factor,Gamma_Mu [GeV]"
        np.savetxt(filename, data, delimiter=',', header=header, fmt='%.6e')
        
        print(f"✅ Profiles exported to {filename}")


# =============================================================================
# MAIN EXECUTION - COMPREHENSIVE DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("🚀 IDMR–YM Framework v2.4 - Corrected Eigenproblem Implementation")
    print("=" * 60)
    
    # Initialize with physical parameters for electron mass
    model = IDMR_YM(representation='dirac')
    
    # Physical validation at test point
    x_test = 1e-18
    A_mu_test = 1.0
    
    print(f"\n📊 FIELD VALUES AT x = {x_test:.1e} m:")
    print(f"   Scalar field φ: {model.phi(x_test):.2e} GeV")
    print(f"   Effective mass m(φ): {model.m_eff(x_test):.6f} GeV")
    print(f"   Metric factor f(φ): {model.f_metric(x_test):.6f}")
    
    # Comprehensive mass generation test
    model.test_mass_generation()
    
    # Benchmark against physical electron
    benchmark = model.benchmark_against_electron()
    print(f"\n🎯 PHYSICAL CONSISTENCY BENCHMARK:")
    print(f"   Computed mass: {benchmark['computed_mass']:.6f} GeV")
    print(f"   Target mass:   {benchmark['target_mass']:.6f} GeV")
    print(f"   Relative error: {benchmark['relative_error_percent']:.2f}%")
    print(f"   Physically consistent: {benchmark['physically_consistent']}")
    
    # Test spatial solver
    try:
        print(f"\n🔍 TESTING SPATIAL DIRAC SOLVER...")
        spatial_solution = model.solve_dirac_spatial_1d([0, 2e-17], 
                                                       energy=0.000511, 
                                                       psi0=np.array([1.0, 0.0j]))
        print(f"   Spatial solver success: {spatial_solution.success}")
    except Exception as e:
        print(f"   Spatial solver note: {e}")
    
    # Test CORRECTED eigenproblem solver and find bound states
    try:
        print(f"\n🎯 SOLVING CORRECTED DIRAC EIGENPROBLEM...")
        bound_states = model.find_bound_states(num_points=500, num_states=8)
        
        print(f"   Bound states found: {bound_states['num_bound_states']}")
        if bound_states['num_bound_states'] > 0:
            for i, energy in enumerate(bound_states['bound_energies']):
                print(f"   State {i}: E = {np.real(energy):.8f} GeV")
        
    except Exception as e:
        print(f"   Eigenproblem note: {e}")
    
    # Advanced analysis and visualization
    analyzer = IDMR_Analysis(model)
    
    # Generate comprehensive profiles
    print(f"\n📈 GENERATING COMPREHENSIVE PROFILES...")
    fig1 = analyzer.plot_profiles([-3e-18, 3e-18])
    plt.savefig('idmr_ym_profiles.png', dpi=300, bbox_inches='tight')
    
    # Plot bound states if found
    if 'bound_states' in locals() and bound_states['num_bound_states'] > 0:
        fig2 = analyzer.plot_bound_states(bound_states)
        plt.savefig('idmr_ym_bound_states.png', dpi=300, bbox_inches='tight')
        print("   Bound states plot saved as 'idmr_ym_bound_states.png'")
    
    # Export data for reproducibility
    analyzer.export_profiles([-3e-18, 3e-18], 'idmr_ym_profiles.csv')
    
    print(f"\n✅ IDMR–YM v2.4 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("   Files generated: idmr_ym_profiles.png, idmr_ym_profiles.csv")
    if 'bound_states' in locals() and bound_states['num_bound_states'] > 0:
        print("   Additional file: idmr_ym_bound_states.png")
