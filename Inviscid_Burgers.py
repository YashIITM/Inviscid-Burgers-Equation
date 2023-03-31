# Import Python dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class domain:

    """
    arguments to the class are as follows:
    1. Number of field cells used in generating the simulation. One less than the number of grid points.
    2. Time step duration used in the simulation.
    3. Velocity of fluid to the left of discontinuity.
    4. Velocity of fluid to the right of discontinuity.
    """

    def __init__(self,no_of_field_cells,time_step_duration,initial_velocity_left,initial_velocity_right,duration=2.0,left_most_grid_point=-1.0,right_most_grid_point = 1.0):
        self.Nx = no_of_field_cells
        self.dt = time_step_duration
        self.i_uL = initial_velocity_left
        self.i_uR = initial_velocity_right
        self.T = duration
        #Number of time steps
        self.Nt = self.T/self.dt
        self.a = left_most_grid_point
        self.b = right_most_grid_point
        #spatial grid length
        self.L = right_most_grid_point-left_most_grid_point
        #time array
        self.time = np.array([range(0,int(self.Nt))]) * self.dt
        #grid spacing and initialize spatial arrays
        self.dx = self.L/self.Nx
        self.x = np.zeros((self.Nx+1,1))
        self.xc = np.zeros((self.Nx,1))
        self.xf = np.zeros((self.Nx+1,1))
        #generate accurate spatial arrays
        for i in range(0,self.Nx+1):
            self.x[i] = i * self.dx + self.a
            self.xf[i] = self.x[i]
        for i in range(0,self.Nx):
            self.xc[i] = 0.5*(self.xf[i] + self.xf[i+1])
        # initialize the state
        u = np.zeros((int(self.Nx),int(self.Nt)), dtype=float)
        for i in range(0,self.Nx):
            if self.xc[i] < self.a + self.L/2:
                 u[i,0] = self.i_uL
            else:
                u[i,0] = self.i_uR
        self.u = u

        # define the flux vector
        self.F = np.zeros((self.Nx+1,1))
        print("Class Initialized...")

    #define flux function for inviscid Burgers' Euation
    @staticmethod
    def flux(u):
        return 0.5*(u**2)
    
    # define the Godunov numerical flux
    @staticmethod
    def GodunovNumericalFlux(uL,uR):
        # compute physical fluxes at left and right state
        FL = domain.flux(uL)
        FR = domain.flux(uR)
        # compute the shock speed
        s = 0.5*(uL + uR)
        # from Toro's book
        if (uL >= uR):
            if (s > 0.0): 
                return FL
            else:
                return FR
        else:
            if (uL > 0.0):
                return FL
            elif (uR < 0.0):
                return FR
            else:
                return 0.0
            
    @staticmethod
    def FSMNumericalFlux(uL,uR):
        # compute physical fluxes at left and right state
        FL = uL*max(0,uL)/2
        FR = uR*min(0,uR)/2
        
        return FL + FR
    
    @staticmethod
    def FDSMNumericalFlux(uL,uR):
        # compute physical fluxes at left and right state
        FL = domain.flux(uL)
        FR = domain.flux(uR)

        u_ = (uL+uR)/2

        flux_FDS = (FL+FR)/2 + (u_/2)*(uL-uR)

        
        return flux_FDS
            
    @staticmethod
    def NumericalFlux(uL,uR,method):
        if method == "gudonov":
            return domain.GodunovNumericalFlux(uL,uR)
        if method == "flux-splitting":
            return domain.FSMNumericalFlux(uL,uR)
        if method == "flux-difference splitting":
            return domain.FDSMNumericalFlux(uL,uR)
        
    def u_exact1(self,x,t):
        if t==0.0:
            if x<=0.0:
                return 1.0
            else:
                return 0.0
        us = (self.i_uL + self.i_uR)/2
        if x<=us*t:
            return 1.0
        else:
            return 0.0
        
    def u_exact2(self,x,t):
        if t==0.0:
            if x<=0.0:
                return 0.0
            else:
                return 1.0
        if x<=0.0:
            return 0.0
        elif x>=1.0*t:
            return 1.0
        else:
            return x/(1.0*t)

    def u_exact3(self,x,t):
        if t==0.0:
            if x<=0.0:
                return -1.0
            else:
                return 1.0
        if x<=self.i_uL*t:
            return -1.0
        elif x>=self.i_uR*t:
            return 1.0
        else:
            return x/(self.i_uR*t)
        
    def u_exact4(self,x,t):
        if t==0.0:
            if x<=0.0:
                return 1.0
            else:
                return -1.0
        us = (1.0 -1.0)/2
        if x<=us*t:
            return 1.0
        else:
            return -1.0
           
    def exact(self):
        # Lets generate the exact solution now:
        if (self.i_uL == 1.0) and (self.i_uR == 0.0):
            uexact = np.zeros((1000,int(self.Nt)))
            x = np.linspace(-1.0,1.0,1000)
            for i in range(1000):
                for t in range(int(self.Nt)):
                    uexact[i,t] = self.u_exact1(x[i] , t * self.dt)
            return uexact
        
        if (self.i_uL == 0.0) and (self.i_uR == 1.0):
            uexact = np.zeros((1000,int(self.Nt)))
            x = np.linspace(-1.0,1.0,1000)
            for i in range(1000):
                for t in range(int(self.Nt)):
                    uexact[i,t] = self.u_exact2(x[i] , t * self.dt)
            return uexact
        
        if (self.i_uL == -1.0) and (self.i_uR == 1.0):
            uexact = np.zeros((1000,int(self.Nt)))
            x = np.linspace(-1.0,1.0,1000)
            for i in range(1000):
                for t in range(int(self.Nt)):
                    uexact[i,t] = self.u_exact3(x[i] , t * self.dt)
            return uexact
        
        if (self.i_uL == 1.0) and (self.i_uR == -1.0):
            uexact = np.zeros((1000,int(self.Nt)))
            x = np.linspace(-1.0,1.0,1000)
            for i in range(1000):
                for t in range(int(self.Nt)):
                    uexact[i,t] = self.u_exact4(x[i] , t * self.dt)
            return uexact
        
        
    
    def GudonovMethod(self):
        # time integrate and obtain the numerical solution using the Gudonov Flux
        for n in range(0,int(self.Nt)-1):
            # estimate the CFL
            CFL = max(abs(self.u[:,n])) * self.dt / self.dx
            if CFL > 0.5:
                print("Warning: CFL > 0.5")

            # compute the interior fluxes
            for i in range(1,self.Nx):
                uL = self.u[i-1,n]
                uR = self.u[i,n]
                self.F[i] = domain.NumericalFlux(uL,uR,"gudonov")

            # compute the left boundary flux
            if self.u[0,n] < 0.0:
                uL = 2.0*self.u[0,n] - self.u[1,n]
            else:
                uL = self.u[0,0]
            uR = self.u[0,n]
            self.F[0] = domain.NumericalFlux(uL,uR,"gudonov")

            # compute the right boundary flux
            if self.u[self.Nx-1,n] > 0.0:
                uR = 2.0 * self.u[self.Nx-1,n] - self.u[self.Nx-2,n]
            else:
                uR = self.u[self.Nx-1,0]
            uL = self.u[self.Nx-1,n]
            self.F[self.Nx] = domain.NumericalFlux(uL,uR,"gudonov")

            # update the state
            for i in range(0,self.Nx):
                self.u[i,n+1] = self.u[i,n] - self.dt / self.dx * (self.F[i+1] - self.F[i])

        uexact = self.exact()

        print('Done! Generating animation...')

        fig = plt.figure()
        ax = plt.axes()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.title("Gudonov Flux Numerical Simulation",size = '10')
        plt.grid()
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlim(-1.0, 1.0)
        if (self.i_uR==0.0) or (self.i_uL==0.0):
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(-1.0, 1.0)

        # plot the initial condition
        ax.plot(np.linspace(-1.0,1.0,1000), uexact[:,0], color='r', linewidth = 1,label = 'Initial Condition')

        # prepare for animated lines
        line, = ax.plot(np.linspace(-1.0,1.0,1000), np.ma.array(np.linspace(-1.0,1.0,1000), mask=True), color='b', linestyle='--', linewidth=2,label = 'Exact Solution')
        line2, = ax.plot(self.xc,np.ma.array(self.xc,mask = True),color = 'g',linestyle='-', linewidth=2, label = 'Numerical Solution')
        def animate(n):
            line.set_ydata(uexact[:,n])
            line2.set_ydata(self.u[:,n])
            return line,line2,

        # Init only required for blitting to give a clean slate.
        def init():
            line.set_ydata(np.ma.array(np.linspace(-1.0,1.0,1000), mask=True))
            line2.set_ydata(np.ma.array(self.xc, mask=True))
            return line,line2,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, int(self.Nt)), init_func=init,
                                    interval=25, blit=True)
        plt.legend()
        plt.show()

    def FluxSplittingMethod(self):
        # time integrate and obtain the numerical solution using the Gudonov Flux
        for n in range(0,int(self.Nt)-1):
            # estimate the CFL
            CFL = max(abs(self.u[:,n])) * self.dt / self.dx
            if CFL > 0.5:
                print("Warning: CFL > 0.5")

            # compute the interior fluxes
            for i in range(1,self.Nx):
                uL = self.u[i-1,n]
                uR = self.u[i,n]
                self.F[i] = domain.NumericalFlux(uL,uR,"flux-splitting")

            # compute the left boundary flux
            if self.u[0,n] < 0.0:
                uL = 2.0*self.u[0,n] - self.u[1,n]
            else:
                uL = self.u[0,0]
            uR = self.u[0,n]
            self.F[0] = domain.NumericalFlux(uL,uR,"flux-splitting")

            # compute the right boundary flux
            if self.u[self.Nx-1,n] > 0.0:
                uR = 2.0 * self.u[self.Nx-1,n] - self.u[self.Nx-2,n]
            else:
                uR = self.u[self.Nx-1,0]
            uL = self.u[self.Nx-1,n]
            self.F[self.Nx] = domain.NumericalFlux(uL,uR,"flux-splitting")

            # update the state
            for i in range(0,self.Nx):
                self.u[i,n+1] = self.u[i,n] - self.dt / self.dx * (self.F[i+1] - self.F[i])

        uexact = self.exact()

        print('Done! Generating animation...')
        
        fig = plt.figure()
        ax = plt.axes()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.title("Flux Splitting Numerical Simulation",size = '10')
        plt.grid()
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlim(-1.0, 1.0)
        if (self.i_uR==0.0) or (self.i_uL==0.0):
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(-1.0, 1.0)

        # plot the initial condition
        ax.plot(np.linspace(-1.0,1.0,1000), uexact[:,0], color='r', linewidth=1,label = 'Initial Condition')

        # prepare for animated lines
        line, = ax.plot(np.linspace(-1.0,1.0,1000), np.ma.array(np.linspace(-1.0,1.0,1000), mask=True), color='b', linestyle='--', linewidth=2,label = 'Exact Solution')
        line2, = ax.plot(self.xc,np.ma.array(self.xc,mask = True),color = 'g',linestyle='-', linewidth=2, label = 'Numerical Solution')
        def animate(n):
            line.set_ydata(uexact[:,n])
            line2.set_ydata(self.u[:,n])
            return line,line2,

        # Init only required for blitting to give a clean slate.
        def init():
            line.set_ydata(np.ma.array(np.linspace(-1.0,1.0,1000), mask=True))
            line2.set_ydata(np.ma.array(self.xc, mask=True))
            return line,line2,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, int(self.Nt)), init_func=init,
                                    interval=25, blit=True)
        plt.legend()
        plt.show()

    def FluxDifferenceSplittingMethod(self):
        # time integrate and obtain the numerical solution using the Gudonov Flux
        for n in range(0,int(self.Nt)-1):
            # estimate the CFL
            CFL = max(abs(self.u[:,n])) * self.dt / self.dx
            if CFL > 0.5:
                print("Warning: CFL > 0.5")

            # compute the interior fluxes
            for i in range(1,self.Nx):
                uL = self.u[i-1,n]
                uR = self.u[i,n]
                self.F[i] = domain.NumericalFlux(uL,uR,"flux-difference splitting")

            # compute the left boundary flux
            if self.u[0,n] < 0.0:
                uL = 2.0*self.u[0,n] - self.u[1,n]
            else:
                uL = self.u[0,0]
            uR = self.u[0,n]
            self.F[0] = domain.NumericalFlux(uL,uR,"flux-difference splitting")

            # compute the right boundary flux
            if self.u[self.Nx-1,n] > 0.0:
                uR = 2.0 * self.u[self.Nx-1,n] - self.u[self.Nx-2,n]
            else:
                uR = self.u[self.Nx-1,0]
            uL = self.u[self.Nx-1,n]
            self.F[self.Nx] = domain.NumericalFlux(uL,uR,"flux-difference splitting")

            # update the state
            for i in range(0,self.Nx):
                self.u[i,n+1] = self.u[i,n] - self.dt / self.dx * (self.F[i+1] - self.F[i])

        uexact = self.exact()

        print('Done! Generating animation...')
        
        fig = plt.figure()
        ax = plt.axes()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.title("Flux Difference Splitting Numerical Simulation",size = '10')
        plt.grid()
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlim(-1.0, 1.0)
        if (self.i_uR==0.0) or (self.i_uL==0.0):
            ax.set_ylim(-0.2, 1.20)
            ax.set_xlim(-1.0, 1.0)

        # plot the initial condition
        ax.plot(np.linspace(-1.0,1.0,1000), uexact[:,0], color='r', linewidth=1,label = 'Initial Condition')

        # prepare for animated lines
        line, = ax.plot(np.linspace(-1.0,1.0,1000), np.ma.array(np.linspace(-1.0,1.0,1000), mask=True), color='b', linestyle='--', linewidth=2,label = 'Exact Solution')
        line2, = ax.plot(self.xc,np.ma.array(self.xc,mask = True),color = 'g',linestyle='-', linewidth=2, label = 'Numerical Solution')
        def animate(n):
            line.set_ydata(uexact[:,n])
            line2.set_ydata(self.u[:,n])
            return line,line2,

        # Init only required for blitting to give a clean slate.
        def init():
            line.set_ydata(np.ma.array(np.linspace(-1.0,1.0,1000), mask=True))
            line2.set_ydata(np.ma.array(self.xc, mask=True))
            return line,line2,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, int(self.Nt)), init_func=init,
                                    interval=25, blit=True)
        plt.legend()
        plt.show()