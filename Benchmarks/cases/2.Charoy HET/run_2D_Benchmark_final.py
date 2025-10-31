from pywarpx import particle_containers, picmi, fields, callbacks
from pywarpx.callbacks import installcallback
from pywarpx import libwarpx as lwx
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import glob, re
import time
"""
When building Python binding library, use:

export WARPX_PRECISION=DOUBLE
export WARPX_PARTICLE_PRECISION=SINGLE
python3 -m pip wheel -v .
"""

"""
Use baseline parameters in Thomas 2025:
initial particle count 75
multigrid precision 1e-3
sorting interval 500
no resampling
"""

"""
Charoy 2019: intermolecular collisions and neutral transport are neglected while a given ionization source term is imposed
"""

class Benchmark_2D_Charoy(object):
#### Constants
    Lx = 2.5e-2
    Ly = 1.28e-2
    Nx = 512 #num of cells
    Ny = 256
    timeStep = 5e-12
    U0 = 200
    ne0 = 5e16
    Te0_eV = 10 # eV
    Ti0_eV = 0.5 # eV
    tmax = 20e-6
    tavg = 16e-6#averaging start time
    nppc0 = 75 #initial particles per cell
    m_i = 2.18e-25

    #for source injection
    S0 = 5.23e23
    x1 = 0.25e-2
    x2 = 1e-2

    # reinject at 2.4cm
    xe = 2.4e-2 #emission plane location

    # averaging steps
    acc_period = 500 #average every xx steps

    #for checkpointing
    # ADD (inside class):
    chk_dir_parent = "chk"     # all WarpX checkpoints live in ./chk/
    chk_prefix     = "chk"     # each folder like chk00010000
    avg_state_file = "avg_state.npz"  # where we store running sums for averaging
    cpPeriod = 50000          # checkpoint period, in steps

    def __init__(self):
        def find_restart_dir():
            # 1) Respect explicit env var if provided
            env = os.environ.get("WARPX_RESTART_DIR")
            if env and os.path.isdir(env):
                return env
            # 2) Otherwise pick latest ./chk/chk*
            base = self.chk_dir_parent
            if not os.path.isdir(base):
                return None
            cands = [d for d in glob.glob(os.path.join(base, f"{self.chk_prefix}*")) if os.path.isdir(d)]
            if not cands:
                return None
            # sort by trailing number (zero-padded) if present
            def key(p):
                m = re.search(r"(\d+)$", os.path.basename(p))
                return int(m.group(1)) if m else os.path.basename(p)
            cands.sort(key=key)
            return cands[-1]
        self.restart_dir = find_restart_dir()

#### Grid definition
        self.grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.Nx, self.Ny],
            lower_bound=[0.0, 0.0],
            upper_bound=[self.Lx, self.Ly],
            lower_boundary_conditions=["dirichlet","periodic"],
            upper_boundary_conditions=["dirichlet","periodic"],
            lower_boundary_conditions_particles=["absorbing","periodic"],
            upper_boundary_conditions_particles=["absorbing","periodic"],
            warpx_potential_lo_x = 200.0,
            warpx_potential_hi_x = 0.0,
        )

        self.Bz = picmi.AnalyticInitialField(
            Bx_expression="0.0",
            By_expression= self.build_Bz_expression(), 
            Bz_expression = "0.0",
            lower_bound = [0.0, 0.0],
            upper_bound = [self.Lx, self.Ly]
        )

#### Field solver
        self.solver = picmi.ElectrostaticSolver(
            grid = self.grid, 
            method="Multigrid",
            required_precision=1e-3,
            #warpx_absolute_tolerance = 1e-3
        )
        
#### Particle types setup, uniform in space, Maxwellian velocity
        self.electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            charge = -picmi.constants.q_e,
            mass = picmi.constants.m_e,
            method = "Boris",
            initial_distribution=picmi.UniformDistribution(
                density=self.ne0,
                rms_velocity=[
                    np.sqrt(self.Te0_eV * picmi.constants.q_e / picmi.constants.m_e),  # vx
                    np.sqrt(self.Te0_eV * picmi.constants.q_e / picmi.constants.m_e),  # vy
                    np.sqrt(self.Te0_eV * picmi.constants.q_e / picmi.constants.m_e)   # vz
                ]
            ),
            warpx_save_particles_at_xlo = True
        )

        #Xenon ion
        self.ions = picmi.Species(
            particle_type="Xe",
            name="ions",
            charge = picmi.constants.q_e,
            mass = self.m_i,
            method = "Boris",
            initial_distribution=picmi.UniformDistribution(
                density=self.ne0,
                rms_velocity=[
                    np.sqrt(self.Ti0_eV * picmi.constants.q_e / self.m_i),  # vx
                    np.sqrt(self.Ti0_eV * picmi.constants.q_e / self.m_i),  # vy
                    np.sqrt(self.Ti0_eV * picmi.constants.q_e / self.m_i)   # vz
                ]
            ),
            warpx_save_particles_at_xlo = True
        )

#### Initialize simulation
        sim_kwargs = dict(
            solver=self.solver,
            time_step_size=self.timeStep,
            max_time=self.tmax,
            particle_shape="linear",
            warpx_field_gathering_algo="energy-conserving",
            warpx_sort_intervals="500",
            verbose=1,
                # --- Domain decomposition parameters ---
            warpx_numprocs=[1, 1],
            warpx_synchronize_velocity = True
        )
        if self.restart_dir:                         # <--- only when we found a checkpoint
            sim_kwargs["warpx_amr_restart"] = self.restart_dir

        self.sim = picmi.Simulation(**sim_kwargs)

        self.sim.add_diagnostic(picmi.Checkpoint(
            period=self.cpPeriod,                         # every 10k steps; tune for your walltime/IO
            write_dir=self.chk_dir_parent,
            warpx_file_prefix=self.chk_prefix,
            warpx_file_min_digits=8
        ))
        

        self.sim.add_species(
            species = self.electrons,
            layout=picmi.PseudoRandomLayout(
                n_macroparticles_per_cell=self.nppc0,
                grid=self.grid
            )
        )
        self.sim.add_species(
            species = self.ions,
            layout=picmi.PseudoRandomLayout(
                n_macroparticles_per_cell=self.nppc0,
                grid=self.grid
            )
        )
        self.sim.add_applied_field(
            self.Bz
        )



#### Compute constants for particle injection
        self.W0 = self.ne0*self.Lx*self.Ly / (self.nppc0*self.Nx*self.Ny)
        self.Ninject0 = int(2*self.S0/np.pi/self.W0*self.Ly*(self.x2-self.x1)*self.timeStep)

    #### Externally applied magnetic field
    
    def build_Bz_expression(self):
        # Parameters
        B0 = 6e-3
        BLx = 1e-3
        Bmax = 10e-3
        xBmax = 0.75e-2
        sigma = 0.625e-2
        Lx = 2.5e-2

        # Calculate coefficients (CORRECTED)
        a1 = (Bmax - B0) / (1 - np.exp(-0.5 * (xBmax/sigma)**2))
        a2 = (Bmax - BLx) / (1 - np.exp(-0.5 * ((Lx - xBmax)/sigma)**2))

        # CORRECTED b1 and b2
        b1 = (B0 - Bmax * np.exp(-0.5 * (xBmax/sigma)**2)) / (1 - np.exp(-0.5 * (xBmax/sigma)**2))
        b2 = (BLx - Bmax * np.exp(-0.5 * ((Lx - xBmax)/sigma)**2)) / (1 - np.exp(-0.5 * ((Lx - xBmax)/sigma)**2))

        # Create the CORRECTED PICMI-parsable expression string
        return f"(x <= {xBmax}) * ({a1} * exp(-(x - {xBmax})**2 / (2 * {sigma}**2)) + {b1}) + (x > {xBmax}) * ({a2} * exp(-(x - {xBmax})**2 / (2 * {sigma}**2)) + {b2})"
    


#### User-defined callbacks
    #### 1. Particle injection
    def sourceInjection(self):
        #inject Ninject0 pairs
        #sample location, in GPU:
        r1 = cp.random.random(self.Ninject0)
        r2 = cp.random.random(self.Ninject0)
        xm = 0.625e-2
        x1 = 0.25e-2
        x2 = 1e-2
        xi = xm + cp.arcsin(2*r1 - 1)*(x2 - x1)/(cp.pi)
        yi = cp.zeros(self.Ninject0)
        zi = r2*self.Ly

        #Box-Muller transform for 3D Maxwellian
        #generate uniform random numbers [0,1)
        u1e = cp.random.random(self.Ninject0)
        u2e = cp.random.random(self.Ninject0)
        u3e = cp.random.random(self.Ninject0)
        u4e = cp.random.random(self.Ninject0)
        u5e = cp.random.random(self.Ninject0)
        u6e = cp.random.random(self.Ninject0)

        u1i = cp.random.random(self.Ninject0)
        u2i = cp.random.random(self.Ninject0)
        u3i = cp.random.random(self.Ninject0)
        u4i = cp.random.random(self.Ninject0)
        u5i = cp.random.random(self.Ninject0)
        u6i = cp.random.random(self.Ninject0)

        #for electrons
        v_th_e = cp.sqrt(self.Te0_eV * picmi.constants.q_e / picmi.constants.m_e)
        uex = v_th_e * cp.sqrt(-2*cp.log(u1e)) * cp.cos(2*cp.pi*u2e)
        uey = v_th_e * cp.sqrt(-2*cp.log(u3e)) * cp.cos(2*cp.pi*u4e)
        uez = v_th_e * cp.sqrt(-2*cp.log(u5e)) * cp.cos(2*cp.pi*u6e)

        #for ions
        v_th_i = cp.sqrt(self.Ti0_eV * picmi.constants.q_e / self.m_i)
        uix = v_th_i * cp.sqrt(-2*cp.log(u1i)) * cp.cos(2*cp.pi*u2i)
        uiy = v_th_i * cp.sqrt(-2*cp.log(u3i)) * cp.cos(2*cp.pi*u4i)
        uiz = v_th_i * cp.sqrt(-2*cp.log(u5i)) * cp.cos(2*cp.pi*u6i)


        particle_containers.ParticleContainerWrapper("electrons").add_particles(
            x = xi.get(),
            y = yi.get(),
            z = zi.get(),
            ux = uex.get(),
            uy = uey.get(),
            uz = uez.get(),
            w = cp.full(self.Ninject0, self.W0).get()
        )

        particle_containers.ParticleContainerWrapper("ions").add_particles(
            x = xi.get(),
            y = yi.get(),
            z = zi.get(),
            ux = uix.get(),
            uy = uiy.get(),
            uz = uiz.get(),
            w = cp.full(self.Ninject0, self.W0).get()
        )

    
    #### 2. Cathode injection
    def cathodeInjection(self):
    #2.1 count number of electrons and ions crossing anode plane
        deltaNe = particle_containers.ParticleBoundaryBufferWrapper().get_particle_boundary_buffer_size(
            species_name="electrons",
            boundary="x_lo",
            local=False
        )
        deltaNi = particle_containers.ParticleBoundaryBufferWrapper().get_particle_boundary_buffer_size(
            species_name="ions",
            boundary="x_lo",
            local = False
        )

        N_reinject = int(deltaNe - deltaNi)
    #2.2 Re-inject deltaNi - deltaNe electrons, uniformly distributed in azimuthal direction
        #velocity is 3D Maxwellian
        if N_reinject > 0:
            r1 = cp.random.random(N_reinject)
            xi = cp.full((N_reinject,), self.xe, dtype=cp.float32) # @Emission plane
            yi = cp.zeros(N_reinject)
            zi = r1*self.Ly

            #Box-Muller transform for 3D Maxwellian
            #generate uniform random numbers [0,1)
            u1 = cp.random.random(N_reinject)
            u2 = cp.random.random(N_reinject)
            u3 = cp.random.random(N_reinject)
            u4 = cp.random.random(N_reinject)
            u5 = cp.random.random(N_reinject)
            u6 = cp.random.random(N_reinject)

            v_th_e = cp.sqrt(self.Te0_eV * picmi.constants.q_e / picmi.constants.m_e)
            uex = v_th_e * cp.sqrt(-2*cp.log(u1)) * cp.cos(2*cp.pi*u2)
            uey = v_th_e * cp.sqrt(-2*cp.log(u3)) * cp.cos(2*cp.pi*u4)
            uez = v_th_e * cp.sqrt(-2*cp.log(u5)) * cp.cos(2*cp.pi*u6)

            particle_containers.ParticleContainerWrapper("electrons").add_particles(
                x = xi.get(),
                y = yi.get(),
                z = zi.get(),
                ux = uex.get(),
                uy = uey.get(),
                uz = uez.get(),
                w = cp.full(N_reinject, self.W0).get()
            )
    #2.3 Clear boundary buffer after each step
        particle_containers.ParticleBoundaryBufferWrapper().clear_buffer()

    #### 3. Potential Adjustment
    def adjustPotential(self):
        phi = np.asarray(fields.PhiFPWrapper(level = 0)[...])  # Use np instead of cp
        Ex = np.asarray(fields.ExFPWrapper(level = 0)[...])
        
        xe_index = int(self.xe / self.Lx * self.Nx)
        Ue = np.mean(phi[xe_index, :])
        
        x_indices = np.arange(phi.shape[0])
        phi_correction = (x_indices / xe_index) * Ue
        phi = phi - phi_correction[:, np.newaxis]
        
        phi_fp = fields.PhiFPWrapper(level = 0)
        phi_fp[...] = phi  # Now it's already NumPy
        
        Ex = Ex + Ue / self.xe
        Ex_fp = fields.ExFPWrapper(level = 0)
        Ex_fp[...] = Ex

    #### 4. Temperature diagnostics
    def temperature_diagnostics(self):
        pc = particle_containers.ParticleContainerWrapper("electrons")

        DT = cp.float32

        # --- helpers: first tile -> CuPy, cast once to DT ---
        def to_cp(a):
            a = a[0] if isinstance(a, (list, tuple)) else a
            return cp.asarray(a, dtype=DT)

        # positions, velocities (SI m/s), weights
        x  = to_cp(pc.get_particle_x(level=0))
        z  = to_cp(pc.get_particle_z(level=0))
        vx = to_cp(pc.get_particle_ux(level=0))   # SI velocity
        vy = to_cp(pc.get_particle_uy(level=0))
        vz = to_cp(pc.get_particle_uz(level=0))
        w  = to_cp(pc.get_particle_weight(level=0))

        # --- manual binning setup: map particles -> (ix, iz) bins ---
        nx, nz = self.Nx + 1, self.Ny + 1
        ix = cp.clip((x * nx / self.Lx).astype(cp.int32), 0, nx - 1)
        iz = cp.clip((z * nz / self.Ly).astype(cp.int32), 0, nz - 1)
        lin = ix * nz + iz

        # --- accumulate sums in one pass: Sw, Σw v, Σw v^2 ---
        v2  = vx*vx + vy*vy + vz*vz
        size = nx * nz
        Sw  = cp.zeros(size, dtype=DT)
        Svx = cp.zeros(size, dtype=DT); Svy = cp.zeros(size, dtype=DT); Svz = cp.zeros(size, dtype=DT)
        Sv2 = cp.zeros(size, dtype=DT)

        cp.add.at(Sw,  lin, w)
        cp.add.at(Svx, lin, w * vx); cp.add.at(Svy, lin, w * vy); cp.add.at(Svz, lin, w * vz)
        cp.add.at(Sv2, lin, w * v2)

        # --- per-cell means and drift removal: <v^2>_th = <v^2> - |<v>|^2 ---
        Sw   = Sw.reshape(nx, nz)
        Svx  = Svx.reshape(nx, nz); Svy = Svy.reshape(nx, nz); Svz = Svz.reshape(nx, nz); Sv2 = Sv2.reshape(nx, nz)

        # Compute averages safely without cp.errstate
        mask = Sw > 0
        vx_bar = cp.zeros_like(Sw)
        vy_bar = cp.zeros_like(Sw)
        vz_bar = cp.zeros_like(Sw)
        v2_bar = cp.zeros_like(Sw)

        # Use elementwise division only where Sw > 0
        vx_bar[mask] = Svx[mask] / Sw[mask]
        vy_bar[mask] = Svy[mask] / Sw[mask]
        vz_bar[mask] = Svz[mask] / Sw[mask]
        v2_bar[mask] = Sv2[mask] / Sw[mask]

        # Fill invalid cells with NaN for plotting convenience
        vx_bar[~mask] = DT(cp.nan)
        vy_bar[~mask] = DT(cp.nan)
        vz_bar[~mask] = DT(cp.nan)
        v2_bar[~mask] = DT(cp.nan)
        vbar2  = vx_bar*vx_bar + vy_bar*vy_bar + vz_bar*vz_bar
        v2_th  = cp.maximum(v2_bar - vbar2, DT(0))  # clip tiny negatives from roundoff

        # --- Te from thermal variance: (1/2) m <v_th^2> = (3/2) kB Te -> Te[eV] = m/(3 q_e) <v_th^2> ---
        me_over_3qe = DT(picmi.constants.m_e / (3.0 * picmi.constants.q_e))
        Te_2D = v2_th * me_over_3qe  # eV

        # --- accumulate time-average on device (move to host only when saving) ---
        if isinstance(self.Te_grid, np.ndarray):
            self.Te_grid += cp.asnumpy(Te_2D)
        else:
            self.Te_grid += Te_2D

#### Diagnostics
    def diagnostics(self):
        istep = int(lwx.warpx.getistep(0))
        if istep % self.acc_period == 0 and istep > 0:
            #rho diagnostics
            phi_data = self.phi_wrapper[...]
            self.phi_array += phi_data

            #Ex diagnostics
            Ex_data = self.Ex_wrapper[...]
            self.Ex_array += Ex_data

            #ni diagnostics
            particle_containers.ParticleContainerWrapper("ions").deposit_charge_density(level=0)
            rho_data = self.rho_wrapper[...]
            self.ni_array += rho_data / picmi.constants.q_e

            self.temperature_diagnostics()

    def saveAverageState(self):
        istep = int(lwx.warpx.getistep(0))
        if istep % self.cpPeriod == 0 and istep > 0:
            # Save running sums to file
            np.savez(self.avg_state_file,
                    Ex_sum=self.Ex_array,
                    phi_sum=self.phi_array,
                    ni_sum=self.ni_array,
                    Te_sum=self.Te_grid)
            print(f"Saved averaging state to {self.avg_state_file}")

#### Run simulation
    def run_sim(self):
        #### Install callbacks for particle injection and potential adjustment
        installcallback('particleinjection', self.sourceInjection)
        installcallback('particleinjection', self.cathodeInjection)
        installcallback('afterEsolve', self.adjustPotential)
        

        #get current step
        self.sim.step(0)
        cur_step = int(lwx.warpx.getistep(0))
        warmup_steps = int(self.tavg / self.timeStep)
        # Calculate number of diagnostic steps
        self.diag_steps = int((self.tmax - self.tavg) / self.timeStep)
        self.total_steps = int(self.tmax/self.timeStep)
        if cur_step <= warmup_steps:
            print("Restarting from step ", cur_step, " / ", self.total_steps, " ,in warmup phase")
            time.sleep(3)
            # Run warmup
            self.sim.step(warmup_steps-cur_step)
            # Create wrapper and get actual shape
            self.Ex_wrapper = fields.ExFPWrapper(0)
            Ex_temp = self.Ex_wrapper[...]
            self.Ex_array = np.zeros_like(Ex_temp) # initialize empty array

            self.phi_wrapper = fields.PhiFPWrapper(0)
            phi_temp = self.phi_wrapper[...]
            self.phi_array = np.zeros_like(phi_temp)

            self.rho_wrapper = fields.RhoFPWrapper(0)
            rho_temp = self.rho_wrapper[...]
            self.ni_array = np.zeros_like(rho_temp)

            self.Te_grid = np.zeros_like(rho_temp)

            # Install diagnostics and run
            print("Run average phase")
            time.sleep(3)
            installcallback('afterstep', self.diagnostics)
            installcallback('afterdiagnostics',self.saveAverageState)
            self.sim.step(self.diag_steps)
            
        else:
            print("Restarting from step ", cur_step, " / ", self.total_steps, " ,after warmup phase")
            time.sleep(3)
            # restoring from diagnostics steps
            # try to reload previous averaging state
            print(f"Loading previous averaging state from {self.avg_state_file}")
            data = np.load(self.avg_state_file)
            self.Ex_array = data['Ex_sum']
            self.phi_array = data['phi_sum']
            self.ni_array = data['ni_sum']
            self.Te_grid = data['Te_sum']
            self.Ex_wrapper = fields.ExFPWrapper(0)
            self.phi_wrapper = fields.PhiFPWrapper(0)
            self.rho_wrapper = fields.RhoFPWrapper(0)

            # Install diagnostics and run
            installcallback('afterstep', self.diagnostics)
            installcallback('afterdiagnostics',self.saveAverageState)
            self.sim.step(self.total_steps - cur_step)
        
        np.save("Ex.npy", self.Ex_array/(self.diag_steps/self.acc_period))
        np.save("phi.npy", self.phi_array/(self.diag_steps/self.acc_period))
        np.save("ni.npy", self.ni_array/(self.diag_steps/self.acc_period))
        np.save("Te.npy", self.Te_grid/(self.diag_steps/self.acc_period))

#### Run sim
run = Benchmark_2D_Charoy()
run.run_sim()


def plot_Results(Exfile, phifile, nifile, Tefile,
                 x_len_cm=2.5, y_len_cm=1.28):
    """
    Read 4 .npy result files and save four figures:
      - Heatmap (x-y) with extent in cm
      - Line plot: y-averaged profile vs x (cm)
    """

    specs = [
        (Exfile,  "Ex",  "Ex [kV/m]",         1e-3),   # V/m -> kV/m
        (nifile,  "ni",  "Number density [m$^{-3}$]", 1.0),
        (phifile, "phi", "Potential [V]",     1.0),
        (Tefile,  "Te",  "Electron temperature [eV]", 1.0),
    ]

    for fname, tag, cbar_label, scale in specs:
        data = np.load(fname)
        plot_data = data * scale

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        extent = [0.0, x_len_cm, 0.0, y_len_cm]
        im = axes[0].imshow(
            plot_data.T, origin='lower', aspect='auto',
            extent=extent, cmap='RdBu_r'
        )
        axes[0].set_xlabel('x (cm)')
        axes[0].set_ylabel('y (cm)')
        axes[0].set_title(f'{tag} field')
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label(cbar_label)

        # ---- averaged line profile ----
        prof = np.nanmean(plot_data, axis=1)
        x_positions = np.linspace(0.0, x_len_cm, len(prof))
        axes[1].plot(x_positions, prof)
        axes[1].set_xlabel('x (cm)')
        axes[1].set_ylabel(cbar_label)
        axes[1].set_title(f'Averaged {tag} Profile')
        axes[1].grid(True, alpha=0.3)

        # ---- set ylim per field ----
        if tag == "Ex":
            axes[1].set_ylim(-5, 60)
            im.set_clim(-5, 60)
        elif tag == "ni":
            axes[1].set_ylim(0, 4.5e17)
            im.set_clim(0, 4.5e17)
        elif tag == "Te":
            axes[1].set_ylim(0, 60)
            im.set_clim(0, 60)
        # (phi left unbounded intentionally)

        plt.tight_layout()

        base = os.path.splitext(os.path.basename(fname))[0]
        out_png = f'{base}_plot.png'
        if os.path.exists(out_png):
            print(f"Warning: {out_png} exists and will be overwritten.")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_png}")


plot_Results("Ex.npy", "phi.npy", "ni.npy", "Te.npy")
