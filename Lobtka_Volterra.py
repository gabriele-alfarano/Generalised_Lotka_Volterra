import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigs
from scipy.stats import linregress
import scipy.fft as fft
from scipy.signal import find_peaks

# 0D ====================================================================================================================
# =======================================================================================================================
# =======================================================================================================================

class LotkaVolterraEco:
    def __init__(self, params):

        self.N = params["N"]
        self.C = params["C"]
        self.eps = params.get("eps", 1e-10)
        self.a = params.get("a", 1.0)
        self.t_end = params["t_end"]

        seed = params.get("seed", 42)
        np.random.seed(seed)
        range_population = params.get("range_population", (0.01, 1))
        self.min_population, self.max_population = range_population

        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.gamma = params["gamma"]
        self.alpha = params["network_structure"]
        self.A = self._build_interaction_matrix()
        
        self.W = (-diags(np.ones(self.N))/self.a + self.A.multiply(self.alpha)).tocsr()
        self.r = np.ones(self.N)
        self.y0 = np.random.uniform(self.min_population, self.max_population, self.N)
        self.t_all, self.y_all = None, None

    # ============================================================
    def _build_interaction_matrix(self):
        alpha_coo = self.alpha.tocoo()
        mask_upper = alpha_coo.row < alpha_coo.col
        rows = alpha_coo.row[mask_upper]
        cols = alpha_coo.col[mask_upper]

        data = []
        row_idx = []
        col_idx = []

        for i, j in zip(rows, cols):
            z1, z2 = np.random.normal(0, 1, 2)

            a_ij = self.mu/self.C + self.sigma/np.sqrt(self.C) * z1
            a_ji = self.mu/self.C + self.sigma/np.sqrt(self.C) * (self.gamma*z1 + np.sqrt(1-self.gamma**2)*z2)

            row_idx.extend([i, j])
            col_idx.extend([j, i])
            data.extend([a_ij, a_ji])

        A_sparse = coo_matrix((data, (row_idx, col_idx)),shape=(self.N, self.N))
        return A_sparse.tocsr()
    
    # ============================================================
    def _model(self, t, y, r, W):
        dy_dt = y * (r + W.dot(y))
        dy_dt[y < self.eps] = 0.0
        return dy_dt

    # ============================================================   
    def run(self):
        def extinction_event(t, y, r, W):
            return np.min(y) - self.eps
        extinction_event.terminal = True
        extinction_event.direction = -1

    # ============================================================
        def stationary_event(t, y, r, W):
            dy_dt = y * (r + W.dot(y))
            return np.max(np.abs(dy_dt)) - 1e-10
        stationary_event.terminal = True
        
        t_current, y_current = 0, self.y0.copy()
        t_hist, y_hist = [], []
        current_indices = np.arange(self.N)
        
        while t_current < self.t_end:
            sol = solve_ivp(self._model, (t_current, self.t_end), y_current, 
                            args=(self.r, self.W), method='LSODA', 
                            events=[extinction_event, stationary_event], atol=1e-8, rtol=1e-6)
            
            t_seg = sol.t if t_current == 0 else sol.t[1:]
            y_seg_reduced = sol.y if t_current == 0 else sol.y[:, 1:]
            
            y_seg_full = np.zeros((self.N, len(t_seg)))
            y_seg_full[current_indices, :] = y_seg_reduced
            
            t_hist.append(t_seg)
            y_hist.append(y_seg_full)
            
            t_current, y_current = sol.t[-1], sol.y[:, -1]
            
            if sol.status == 1: # t_events[0] -> extinction, t_events[1] -> stationary

                if sol.t_events[0].size > 0:
                    alive_mask = y_current > (self.eps + 1e-15)
                    if not np.any(alive_mask): 
                        break 
                    
                    current_indices = current_indices[alive_mask]
                    y_current = y_current[alive_mask]
                    self.r = self.r[alive_mask]
                    self.W = self.W[alive_mask, :][:, alive_mask]
                    # Continua il ciclo while per ricalcolare la dinamica
                
                elif sol.t_events[1].size > 0:
                    print(f"Stazionarietà raggiunta al tempo {t_current:.2f}")
                    break
            else:
                break

        self.surviving_indices = current_indices
        self.surviving_network = self.alpha[self.surviving_indices, :][:, self.surviving_indices]
        self.phi = len(self.surviving_indices) / self.N
        self.t_all = np.concatenate(t_hist)
        self.y_all = np.concatenate(y_hist, axis=1)

    # ============================================================
    def plot_eigenvalues(self, return_eigenvalues = False):
        
        y_alive = self.y_all[:, -1][self.surviving_indices]
        W_alive = self.W 

        R = diags(y_alive)
        RW = (R @ W_alive).toarray()

        evals = np.linalg.eigvals(RW)
        max_re = np.max(evals.real)
        min_re = np.min(evals.real)

        plt.figure(figsize=(9, 8))
        plt.scatter(evals.real, evals.imag, s=20, alpha=0.5, color='teal', marker='x')
        plt.axvline(0, color='red', linestyle='--', alpha=0.6)
        plt.axvline(max_re, color='orange', lw=2, label=f'Max Re(λ) = {max_re:.4f}')
        plt.axvline(min_re, color='green', lw=2, label=f'Min Re(λ) = {min_re:.4f}')

        nmax = np.max(y_alive)
        M = np.sum(y_alive) / self.N
        phi = phi = len(y_alive) / self.N
        mean_field = -nmax * (0.5 + 0.5*np.sqrt(1+ 4 * M * phi *(self.gamma*self.sigma**2 + self.mu**2/self.C*(1-self.C/self.N)) / (phi * nmax - M))) 
        mean_field_fully_connected = -nmax * (0.5 + 0.5*np.sqrt(1+ 4 * M * phi *(self.gamma*self.sigma**2) / (phi * nmax - M))) 

        plt.plot(-1, 0, marker='v', linestyle='None',
                markerfacecolor='none',
                markeredgecolor='orange', markeredgewidth=2,
                markersize=10, label='-1 (outlier)')
        
        plt.plot(-1-self.mu*M, 0, marker='v', linestyle='None',
                markerfacecolor='none',
                markeredgecolor='green', markeredgewidth=2,
                markersize=10, label=r'$-1-\mu M (outlier)$')

        plt.plot(-nmax, 0, marker='o', linestyle='None',
                markerfacecolor='none',
                markeredgecolor='purple', markeredgewidth=2,
                markersize=10, label='-nmax')

        plt.plot(mean_field, 0, marker='^', linestyle='None',
                markerfacecolor='none',
                markeredgecolor='red', markeredgewidth=2,
                markersize=10, label='Mean field')
        
        plt.plot(mean_field_fully_connected, 0, marker='^', linestyle='None',
                markerfacecolor='none',
                markeredgecolor='violet', markeredgewidth=2,
                markersize=10, label='Mean field fully connected')

        plt.title(f"RW spectrum for surviving species (N'={len(self.surviving_indices)})")
        plt.xlabel("Re(λ)")
        plt.ylabel("Im(λ)")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.show()
        if return_eigenvalues:
            return evals

    # ============================================================
    def plot_summary(self, degrees=None):

        if degrees is not None:
            connectivity = np.array(degrees)
        else:
            connectivity = np.asarray(self.alpha.sum(axis=1)).ravel()
        final_biomass = self.y_all[:, -1]
        
        t = self.t_all
        is_uniform = np.ptp(connectivity) == 0 if len(connectivity) > 0 else True

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.cm.viridis

        if is_uniform:
            val = connectivity[0] if len(connectivity) > 0 else 0
            norm = mpl.colors.Normalize(vmin=val - 1, vmax=val + 1)
        else:
            norm = mpl.colors.Normalize(vmin=connectivity.min(), vmax=connectivity.max())

        # --- SUBPLOT 1: Dinamica temporale -------------------
        
        for i in range(len(connectivity)):
            color = cmap(norm(connectivity[i]))
            ax1.plot(t, self.y_all[i, :], lw=0.8, color=color, alpha=0.6)
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel(r"Biomass $\rho_i$")
        ax1.set_title(fr"Dinamica Specie ($\phi$={self.phi:.3f})")
        ax1.grid(True, which="both", ls="-", alpha=0.15)

        # --- SUBPLOT 2: Biomassa vs Connettività --------------

        ax2.scatter(connectivity, final_biomass, c=connectivity, cmap=cmap, 
                    norm=norm, alpha=0.7, edgecolors='k', linewidths=0.5, s=30, label="Specie")

        if not is_uniform and len(connectivity) > 1:
            slope, intercept, r_value, _, _ = linregress(connectivity, final_biomass)
            x_range = np.linspace(connectivity.min(), connectivity.max(), 100)
            y_fit = slope * x_range + intercept
            label_text = fr"$\rho_k = {slope:.4f} \cdot k + {intercept:.4f}$" + "\n" + fr"$R^2 = {r_value**2:.3f}$"
            ax2.plot(x_range, y_fit, color="red", lw=2, label=label_text)

            slope = self.mu / self.C / (1 - self.mu * np.mean((connectivity/self.C)**2))
            ax2.plot(x_range, 1 + slope * x_range , color="blue", lw=2, label="theoretical prediction")
            ax2.legend(loc='best', fontsize='small')
            ax2.set_title("Correlazione Post-Selezione")
        else:
            ax2.set_title("Connettività Uniforme o Dati Insufficienti")
            if len(connectivity) > 0:
                ax2.set_xlim(connectivity[0] - 1, connectivity[0] + 1)

        ax2.set_xlabel(r"Connectivity ($k_i$)")
        ax2.set_ylabel(r"Final Biomass $\rho_i$")
        ax2.grid(True, which="both", ls="-", alpha=0.15)

        # Colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax2, label='Connectivity (degree k)')

        plt.tight_layout()
        plt.show()

# 1D ====================================================================================================================
# =======================================================================================================================
# =======================================================================================================================

class SpatialLotkaVolterra:
    def __init__(self, params):

        self.N = params["N"]                    # Number of species  
        self.C = params["C"]

        self.L=params["L"]                      # Spatial domain size
        self.Q=params["Q"]                      # Number of mesh points
        self.h=self.L/self.Q                    # Mesh spacing
        self.delta=params["delta"]              # Absolute size of the kernel (delta < L)

        self.D=params["D"]                      # Diffusion coefficient
        self.eps = params.get("eps", 1e-10)
        self.a = params.get("a", 1.0)

        self.dt=params["dt"]                    # Time step
        self.iterations = params.get("iterations", 2000)
        self.plotStep = params.get("plotStep", 100)

        seed = params.get("seed", 42)
        self.rng = np.random.default_rng(seed)
        range_population = params.get("range_population", (0.01, 1))
        self.min_population, self.max_population = range_population

        self.mu = params["mu"]
        self.sigma = params["sigma"]
        self.gamma = params["gamma"]
        self.alpha = params["network_structure"]
        self.A = self._build_interaction_matrix()
        self.W = (-diags(np.ones(self.N))/self.a + self.A.multiply(self.alpha)).tocsr()
        
        self.rho = self._init_density_field()
        self.LinOp = self._build_linear_operator()
        self.doorKernel = self.build_kernel()

        # --- Output containers ---
        self.abundances = None
        self.data_rho = None
        self.phi = None
        self.alive = None

    # ============================================================
    def _build_interaction_matrix(self):
        alpha_coo = self.alpha.tocoo()
        mask_upper = alpha_coo.row < alpha_coo.col
        rows = alpha_coo.row[mask_upper]
        cols = alpha_coo.col[mask_upper]

        data = []
        row_idx = []
        col_idx = []

        for i, j in zip(rows, cols):
            z1, z2 = np.random.normal(0, 1, 2)

            a_ij = self.mu/self.C + self.sigma/np.sqrt(self.C) * z1
            a_ji = self.mu/self.C + self.sigma/np.sqrt(self.C) * (self.gamma*z1 + np.sqrt(1-self.gamma**2)*z2)

            row_idx.extend([i, j])
            col_idx.extend([j, i])
            data.extend([a_ij, a_ji])

        A_sparse = coo_matrix((data, (row_idx, col_idx)),shape=(self.N, self.N))
        return A_sparse.tocsr()

    # ============================================================
    def _init_density_field(self):
        rho0 = np.ones((self.N, self.Q)) * (1 + self.rng.normal(0, 0.15, size=(self.N, self.Q)))
        return rho0
    
    # ============================================================
    def _build_linear_operator(self):
        k = 2 * np.pi * fft.rfftfreq(self.Q, d=self.h) # Frequenze angolari k = 2*pi*n / L
        k_sq = k**2
        # Operatore semi-implicito: (1 + D * dt * k^2)^-1
        factor = 1.0 / (1.0 + self.D * self.dt * k_sq)
        return np.tile(factor, (self.N, 1))
    
    # ============================================================
    def build_kernel(self):
        doorKernel = np.zeros(self.Q)
        sizeKernel = int(self.delta / self.h)
        idx = np.arange(sizeKernel+1)
        doorKernel[idx] = 1.0
        doorKernel[-idx] = 1.0
        doorKernel /= doorKernel.sum()
        return doorKernel
        
    # ============================================================   
    def _simulate(self):
        n_saves = self.iterations // self.plotStep + 1
        data_abundances = np.empty((self.N, n_saves))
        data_rho = np.empty((self.N, self.Q, n_saves))
        
        c_shape = (self.N, self.Q // 2 + 1)
        rho_F = np.empty(c_shape, dtype=np.complex128)
        aux_F = np.empty(c_shape, dtype=np.complex128)
        
        L_Op = self.LinOp[:, :self.Q // 2 + 1]
        self.k_F = fft.rfft(self.doorKernel)
        rho = self.rho
        W = self.W
        dt = self.dt
        kw = {'axis': 1, 'workers': -1}
        self.alive = np.ones(self.N, dtype=bool)

        for c in range(self.iterations + 1):
            # 0. Forza a zero le specie estinte (sicurezza numerica)
            rho[~self.alive, :] = 0.0

            # 1. Convoluzione del campo di interazione
            inter = W.dot(rho) + 1.0 
            aux_F[:] = fft.rfft(inter, **kw)
            aux_F *= self.k_F
            nonlin = fft.irfft(aux_F, n=self.Q, **kw)
            
            # 2. Step spettrale semi-implicito
            rhs = rho * nonlin
            aux_F[:] = fft.rfft(rhs, **kw)
            rho_F[:] = fft.rfft(rho, **kw)
            
            rho_F += dt * aux_F
            rho_F *= L_Op
            rho[:] = fft.irfft(rho_F, n=self.Q, **kw)
            np.maximum(rho, 0, out=rho)
            
            # 3. Floor di estinzione e aggiornamento maschera
            abundances = np.zeros(self.N)
            abundances[self.alive] = rho[self.alive, :].sum(axis=1) * self.h / self.L
            newly_extinct = self.alive & (abundances < self.eps)
            self.alive[newly_extinct] = False
            rho[~self.alive, :] = 0.0

            # 4. Salvataggio dati
            if c % self.plotStep == 0:
                s_idx = c // self.plotStep
                data_abundances[:, s_idx] = abundances
                data_rho[:, :, s_idx] = rho.copy()
        
        return data_abundances, data_rho
    
    # ============================================================
    def run(self):
        self.abundances, self.data_rho = self._simulate()
        final_abundances = self.abundances[:, -1]
        self.phi = np.mean(final_abundances > self.eps)

        return self

    # ============================================================
    def plot_biomass(self, degrees = None):
        t = np.arange(self.abundances.shape[1]) * self.dt * self.plotStep
        if degrees is not None:
            connectivity = np.array(degrees)
        else:
            connectivity = np.asarray(self.alpha.sum(axis=1)).ravel()
        final_biomass = self.abundances[:, -1]
        is_uniform = np.ptp(connectivity) == 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.cm.viridis

        # Fix Norm per far coincidere i colori
        if is_uniform:
            val = connectivity[0]
            norm = mpl.colors.Normalize(vmin=val - 1, vmax=val + 1)
        else:
            norm = mpl.colors.Normalize(vmin=connectivity.min(), vmax=connectivity.max())

        # --- SUBPLOT 1: Dinamica temporale ---
        for i in range(self.N):
            color = cmap(norm(connectivity[i]))
            ax1.plot(t, self.abundances[i], lw=0.7, color=color, alpha=0.5)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(r"Biomass $\rho_i$")
        ax1.set_title(rf"Dinamica Temporale ($\phi={self.phi:.3f}$)")
        ax1.grid(True, which="both", ls="-", alpha=0.15)

        # --- SUBPLOT 2: Biomassa vs Connettività ---
        ax2.scatter(connectivity, final_biomass, c=connectivity, cmap=cmap, 
                    norm=norm, alpha=0.7, edgecolors='none', s=20, label="Specie")

        if not is_uniform:
            slope, intercept, r_value, _, _ = linregress(connectivity, final_biomass)
            x_range = np.linspace(connectivity.min(), connectivity.max(), 100)
            y_fit = slope * x_range + intercept
            label_text = fr"$\rho_k = {slope:.4f} \cdot k + {intercept:.4f}$" + "\n" + fr"$R^2 = {r_value**2:.3f}$"
            ax2.plot(x_range, y_fit, color="red", lw=2, label=label_text)
            ax2.legend(loc='best', fontsize='small')
            ax2.set_title("Correlazione Biomassa-Connettività")
        else:
            ax2.set_title("Connettività Uniforme")
            ax2.set_xlim(connectivity[0] - 1, connectivity[0] + 1)

        ax2.set_xlabel(r"Connectivity ($k_i$)")
        ax2.set_ylabel(r"Final Biomass $\rho_i$")
        ax2.grid(True, which="both", ls="-", alpha=0.15)

        # Colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax2, label='Connectivity')
        if is_uniform:
            cb.set_ticks([connectivity[0]])

        plt.tight_layout()
        plt.show()
        
    # ============================================================
    def plot_spatial_profiles(self, degrees = None):
        x = np.linspace(0, self.L, self.Q)
        rho_final = self.data_rho[:, :, -1]
        if degrees is not None:
            connectivity = np.array(degrees)
        else:
            connectivity = np.array(self.alpha.sum(axis=1)).flatten()

        mask_alive = self.alive
        has_extinction = np.any(~mask_alive)
        is_uniform = np.ptp(connectivity) == 0

        # Normalizzazione colori plot originale
        if is_uniform:
            val = connectivity[0]
            norm = mpl.colors.Normalize(vmin=val - 1, vmax=val + 1)
        else:
            norm = mpl.colors.Normalize(
                vmin=connectivity.min(),
                vmax=connectivity.max() + 1e-12
            )

        cmap = plt.cm.viridis

        if has_extinction:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 8))

        # SUBPLOT 1: COMUNITÀ ORIGINALE
        for i in range(rho_final.shape[0]):
            ax1.plot(
                x,
                rho_final[i, :],
                color=cmap(norm(connectivity[i])),
                lw=0.5,
                alpha=0.3,
                zorder=1
            )

        ax1.set_title("Spatial profiles at final time: Community overview", fontsize=14)
        ax1.set_xlabel("Position $x$")
        ax1.set_ylabel(r"Density $\rho_i(x)$")
        ax1.grid(alpha=0.2, linestyle='--')

        sm1 = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm1.set_array([])
        cb1 = fig.colorbar(sm1, ax=ax1, label='Connectivity')

        if is_uniform:
            cb1.set_ticks([connectivity[0]])

        # SUBPLOT 2: SOLO SOPRAVVISSUTE (se ci sono estinzioni)
        
        if has_extinction:
            new_alpha = self.alpha.tocsr()[mask_alive][:, mask_alive]
            new_connectivity = np.asarray(new_alpha.sum(axis=1)).ravel()

            cmap2 = plt.cm.plasma
            is_uniform_new = np.ptp(new_connectivity) == 0

            if is_uniform_new:
                val = new_connectivity[0]
                norm2 = mpl.colors.Normalize(vmin=val - 1, vmax=val + 1)
            else:
                norm2 = mpl.colors.Normalize(
                    vmin=new_connectivity.min(),
                    vmax=new_connectivity.max() + 1e-12
                )

            alive_idx = np.where(mask_alive)[0]
            for j, species in enumerate(alive_idx):
                ax2.plot(
                    x,
                    rho_final[species, :],
                    color=cmap2(norm2(new_connectivity[j])),
                    lw=1.2,
                    alpha=0.7
                )

            ax2.set_title("Surviving species: renormalized connectivity", fontsize=14)
            ax2.set_xlabel("Position $x$")
            ax2.set_ylabel(r"Density $\rho_i(x)$")
            ax2.grid(alpha=0.2, linestyle='--')

            sm2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
            sm2.set_array([])
            cb2 = fig.colorbar(sm2, ax=ax2, label='New Connectivity')

            if is_uniform_new:
                cb2.set_ticks([new_connectivity[0]])

        plt.tight_layout()
        plt.show()

        # Analisi lunghezze d'onda
        all_wavelengths = []
        for species in range(rho_final.shape[0]):
            peaks, _ = find_peaks(rho_final[species, :])
            if len(peaks) >= 2:
                all_wavelengths.extend(np.diff(x[peaks]))

        if len(all_wavelengths) > 0:
            k_values = 2 * np.pi / np.array(all_wavelengths)
            print(f"--- Analisi spaziale ---")
            print(f"Vettore d'onda critico (kc): {np.mean(k_values):.4f} ± {np.std(k_values):.4f}")

    # ============================================================
    def abundance_vs_weighted_mean_spatial(self, degrees = None):
        x = np.linspace(0, self.L, self.Q)

        rho = self.data_rho[:, :, -1]
        alpha_csr = self.alpha.tocsr()

        if degrees is not None:
            connectivity = np.array(degrees)
        else:
            connectivity = np.array(alpha_csr.sum(axis=0)).flatten()

        weights = connectivity / self.C
        weighted_mean = (weights[:, None] * rho).sum(axis=0) / self.N

        unique_k = np.unique(connectivity)

        is_uniform = np.ptp(connectivity) == 0
        if is_uniform:
            val = connectivity[0]
            norm = mpl.colors.Normalize(vmin=val - 1, vmax=val + 1)
        else:
            norm = mpl.colors.Normalize(vmin=connectivity.min(),
                                        vmax=connectivity.max() + 1e-12)

        cmap = plt.cm.viridis

        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.05], height_ratios=[2, 1, 0.15])

        ax_main = fig.add_subplot(gs[0, 0:2])
        ax_ratio = fig.add_subplot(gs[1, 0])
        ax_fit = fig.add_subplot(gs[1, 1])

        ratio_mean_vs_k = []
        ratio_dict = {}

        ax_main.plot(x, weighted_mean, color='black', lw=3, label='Weighted mean')

        for k in unique_k:
            idx = np.where(connectivity == k)[0]
            if len(idx) == 0:
                continue

            mean_k = rho[idx, :].mean(axis=0)

            ax_main.plot(x, mean_k,
                        color=cmap(norm(k)),
                        lw=1.5,
                        alpha=0.8)

            ratio = mean_k / (weighted_mean + 1e-12)
            ratio_dict[k] = ratio
            ratio_mean_vs_k.append(np.mean(ratio))

        ax_main.set_title("Weighted mean vs same-k averages")
        ax_main.grid(alpha=0.2)

        for k in unique_k:
            if k not in ratio_dict:
                continue
            ax_ratio.plot(x, ratio_dict[k],
                        color=cmap(norm(k)),
                        lw=1.2,
                        alpha=0.8)

        ax_ratio.set_title("Ratio vs x")
        ax_ratio.grid(alpha=0.2)

        k_vals = unique_k
        ratio_mean_vs_k = np.array(ratio_mean_vs_k)

        if len(k_vals) > 1:
            slope, intercept = np.polyfit(k_vals, ratio_mean_vs_k, 1)
            print("slope = ", slope)
            print("intercept = ", intercept)
            r_value = np.corrcoef(k_vals, ratio_mean_vs_k)[0, 1]

            k_fit = np.linspace(k_vals.min(), k_vals.max(), 100)
            y_fit = slope * k_fit + intercept

            ax_fit.scatter(k_vals, ratio_mean_vs_k, color='black')
            label_text = fr"$y = {slope:.3f}k + {intercept:.3f}$" + "\n" + fr"$R^2 = {r_value**2:.3f}$"
            ax_fit.plot(k_fit, y_fit, color='red', label = label_text)
            a = 1 - self.mu * np.mean((connectivity/self.C)**2)
            ax_fit.plot(k_fit, a + self.mu * (k_fit / self.C), color='green', label = "theoretical predictions")

        ax_fit.set_title("Mean ratio vs k")
        ax_fit.grid(alpha=0.2)
        ax_fit.legend()

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_subplot(gs[2, 0:2])
        fig.colorbar(sm, cax=cax, orientation='horizontal')

        plt.tight_layout()
        plt.show()