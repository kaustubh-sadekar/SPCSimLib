import torch


class PostProcEDH:
    r"""Class for post processing the EDH data for distance estimation.
    """
    def __init__(self, Nr, Nc, N_tbins, tmax, device):
        r"""Set up the indexing arrays and common tensors for the class

        Args:
            Nr (int): Number of rows
            Nc (int): Number of columns
            device (str): Device `cpu` or `gpu`
            N_tbins (int): Number of time bins (frame)
        """
        
        self.Nr = Nr
        self.Nc = Nc
        self.N_tbins = N_tbins
        self.device = device
        self.rr,self.cc  = torch.meshgrid(torch.linspace(0,self.Nr-1,self.Nr), 
                                          torch.linspace(0,self.Nc-1,self.Nc), indexing='ij')
        self.rr = self.rr.to(device = self.device,
                                            dtype = torch.long)
        self.cc = self.cc.to(device = self.device,
                                            dtype = torch.long)
        self.tmax = tmax
        

    def interp_nonuni_t(self, x1_,y1_,N, mode=1):
        r"""This function takes the x and y data values as 3D tensors and interpolates the y values along ``N`` equally spaced x values

        Args:
            x1_ (torch.tensor): Location of the mid-points of ED histogram bins. Tensor shape (Nr, Nc, <Num EDH bins>)
            y1_ (torch.tensor): Photon density estimate for each ED histogram bins (Inverse values of the ED histogram bin widths). Tensor shape (Nr, Nc, <Num EDH bins>)
            N (int): Number of equally spaced time bins to interpolate the photon density estimates
            mode (int, optional): Choose mode=0 for NN interpolation and mode = 1 for linear interpolation. Defaults to 1.

        Returns:
            x2 (torch.tensor): Location of the equally spaced interpolated mid-points of ED histogram bins. Tensor shape (Nr, Nc, N)
            y2 (torch.tensor): Value of the the interpolated photon density estimates of ED histogram bins. Tensor shape (Nr, Nc, N)
        """
        
        Nr, Nc = x1_.shape[:2]
        x2 = torch.linspace(0,self.N_tbins-1,N).repeat((Nr,Nc,1)).to(self.device)
        
        y2 = x2*0.0
        y2[:,:,-1] = y1_[:,:,-1]


        if mode==0:

            for i in range(x1_.shape[-1]-1):
                mask1_ = x2>= x1_[:,:,i].reshape(x1_.shape[0],x1_.shape[1],1)
                mask3_ = x2<x1_[:,:,i+1].reshape(x1_.shape[0],x1_.shape[1],1)
                mask_ = mask1_*mask3_
                x_,y_,z_ = torch.nonzero(mask_, as_tuple=True)
                y2[x_,y_,z_] = y1_[x_,y_,i+1].to(torch.float)

        elif mode==1:

            for i in range(x1_.shape[-1]-1):
                mask1_ = x2>= x1_[:,:,i].reshape(x1_.shape[0],x1_.shape[1],1)
                mask3_ = x2<x1_[:,:,i+1].reshape(x1_.shape[0],x1_.shape[1],1)
                mask_ = mask1_*mask3_
                x_,y_,z_ = torch.nonzero(mask_, as_tuple=True)
                m_ = (y1_[x_,y_,i+1] - y1_[x_,y_,i])/(x1_[x_,y_,i+1] - x1_[x_,y_,i] + 0.00000000000001)
                y2[x_,y_,z_] = (m_*(x2[x_,y_,z_] - x1_[x_,y_,i+1]) + y1_[x_,y_,i+1]).to(torch.float)
        else:
            print("Choose mode=0 for NN interpolation and mode = 1 for linear interpolation")
            exit(-1)

        return x2,y2

    def edh2depth_t(self, eqbins_t, mode=0):
        r""" This is the latest depth estimation method that includes three different depth estimation methods.

        mode = 0: Rho_0 distance estimation after resampling to ``N_tbins`` number of equally spaced EDH values obtained using nearest neighbor interpolation.
        mode = 1: Rho_1 distance estimation after resampling to ``N_tbins`` number of equally spaced EDH values obtained using  linear interpolation.
        mode = 2: Naive distance estimation using narrowest bin of the EDH without any interpolation
        """

        # In case of using just a median tracking binner the median value is returned as the distance estimate for all the modes
        if eqbins_t.shape[-1] == 1:
            depth_idx_ = eqbins_t[:,:,0]
            tof_ = (depth_idx_*1.0/self.N_tbins)*self.tmax
            depth_ = tof_*3e8*0.5*1e-9
            return 0, 0, depth_idx_, depth_  

        
        e_= (eqbins_t*1.0).to(torch.double)
        
        bin_w_ = torch.abs(torch.diff(e_, axis=-1))
        bin_w_inv_ = 1.0/(bin_w_+0.0000000001)
        bin_idx_ = (e_[:,:,:-1] + 0.5*bin_w_ -0.5)

        # print("torch.zeros((self.Nr, self.Nc,1), device=self.device)",torch.zeros((self.Nr, self.Nc,1), device=self.device))
        # print("bin_idx_", bin_idx_)
        # print("torch.ones((self.Nr, self.Nc,1), device=self.device)*self.N_tbins)", torch.ones((self.Nr, self.Nc,1), device=self.device)*self.N_tbins)
        
        bin_idx_ = torch.concat((torch.zeros((self.Nr, self.Nc,1), device=self.device), bin_idx_, torch.ones((self.Nr, self.Nc,1), device=self.device)*self.N_tbins), -1)
        bin_w_inv_ = torch.cat((bin_w_inv_[:,:,0].view(self.Nr,self.Nc,1), bin_w_inv_, bin_w_inv_[:,:,-1].view(self.Nr,self.Nc,1)), -1)

        if mode == 0:
            bin_idx_, bin_w_inv_ = self.interp_nonuni_t(bin_idx_,bin_w_inv_,self.N_tbins, mode=0)
            
            idx_temp_ = torch.argmax(bin_w_inv_, axis=2).cpu().numpy()
            depth_idx_ = torch.tensor(bin_idx_.cpu().numpy()[self.rr.cpu().numpy(),self.cc.cpu().numpy(),idx_temp_[self.rr.cpu().numpy(),self.cc.cpu().numpy()]])
            
            tof_ = (depth_idx_*1.0/self.N_tbins)*self.tmax

            depth_ = tof_*3e8*0.5*1e-9
            
        elif mode == 1:
            bin_idx_, bin_w_inv_ = self.interp_nonuni_t(bin_idx_,bin_w_inv_,self.N_tbins, mode=1)
            
            idx_temp_ = torch.argmax(bin_w_inv_, axis=2).cpu().numpy()
            depth_idx_ = torch.tensor(bin_idx_.cpu().numpy()[self.rr.cpu().numpy(),self.cc.cpu().numpy(),idx_temp_[self.rr.cpu().numpy(),self.cc.cpu().numpy()]])
            
            tof_ = (depth_idx_*1.0/self.N_tbins)*self.tmax

            depth_ = tof_*3e8*0.5*1e-9
        
        elif mode ==2:
            idx_temp_ = torch.argmax(bin_w_inv_, axis=2)
            depth_idx_ = bin_idx_[self.rr.cpu().numpy(),
                                  self.cc.cpu().numpy(),
                                  idx_temp_[self.rr.cpu().numpy(),
                                            self.cc.cpu().numpy()].cpu().numpy()]
            tof_ = (depth_idx_.clone().to(torch.float)/self.N_tbins)*self.tmax*1e-9
            depth_ = tof_*3e8*0.5
            
        else:
            print("Incorrect mode selected!! choose 0 or 1")
            exit(-1)


        return bin_w_inv_, bin_idx_, depth_idx_, depth_