def fun_GSoperator_NN_conv_smooth_batch_adaptive(
    f,
    Laplace_kernel_ds,
    Df_dr_kernel_ds,
    RR_ds,
    ZZ_ds,
):

    f = tf.transpose(f, [3, 1, 2, 0])
    Lpsi = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Laplace_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Lpsi = tf.transpose(
        Lpsi, [3, 1, 2, 0]
    )  # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    Dpsi_dr = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Df_dr_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Dpsi_dr = (
        -Dpsi_dr
    )  # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = tf.transpose(Dpsi_dr, [3, 1, 2, 0])
    RR_in = tf.expand_dims(RR_ds[:, 1:-1, 1:-1], axis=-1)
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in)

    GS_ope = Lpsi - Dpsi_dr

    hr = RR_ds[:, 1, 2] - RR_ds[:, 1, 1]
    hz = ZZ_ds[:, 2, 1] - ZZ_ds[:, 1, 1]
    alfa = -2 * (hr**2 + hz**2)

    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr, axis=-1), axis=-1), axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz, axis=-1), axis=-1), axis=-1)
    alfa = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(alfa, axis=-1), axis=-1), axis=-1
    )

    GS_ope = GS_ope * alfa / (hr**2 * hz**2)
    # GS_ope[0,:10,0]

    GS_ope = tf.nn.conv2d(GS_ope, Gauss_tensor, strides=[1, 1, 1, 1], padding="SAME")
    GS_ope = tf.squeeze(GS_ope, axis=-1)

    # GS_ope_padded = RHS_ds.numpy()
    # GS_ope_padded[:,1:-1,1:-1] = GS_ope
    # GS_ope = tf.constant(GS_ope_padded)
    return GS_ope


def loss_fun_MSE(y_ds, predictions):
    loss_MSE = tf.reduce_mean(tf.square(y_ds - tf.squeeze(predictions)))
    return loss_MSE


def loss_fun_PDE_adaptive(
    predictions, RHS_in_ds, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
):
    GS_ope_ref = RHS_in_ds
    GS_ope_ds = fun_GSoperator_NN_conv_smooth_batch_adaptive(
        predictions, Laplace_kernel_ds, Df_dr_kernel_ds, RR_ds, ZZ_ds
    )
    loss_PDE = tf.reduce_mean(tf.square(GS_ope_ref - GS_ope_ds))
    return 0.1 * loss_PDE


import torch.nn.functional as F


def _compute_grad_shafranov_operator(
    pred: Tensor,
    Laplace_kernel: Tensor,
    Df_dr_kernel: Tensor,
    RR: Tensor,
    ZZ: Tensor,
    Gauss_kernel: Tensor,
) -> Tensor:
    """
    This implementation is taken from the paper https://doi.org/10.1016/j.fusengdes.2024.114193,
    cited as [1] in the code
    """

    batch_size = pred.shape[0]

    # Compute the Laplace and the Dpsi_dr operators
    Lpsi = F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Laplace_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)

    # The '-' is necessary because the depthwise conv filtershas to be transposed
    # to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = -F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Df_dr_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)
    Dpsi_dr = torch.div(Dpsi_dr, RR[:, None, 1:-1, 1:-1])

    # the LHS in equation 12 of [1]
    lhs = Lpsi - Dpsi_dr

    # GS operator (RHS of requation 12 of [1])
    hr = (RR[:, 1, 2] - RR[:, 1, 1])[:, None, None, None]
    hz = (ZZ[:, 2, 1] - ZZ[:, 1, 1])[:, None, None, None]
    alfa = -2 * (hr**2 + hz**2)
    GS_ope = lhs * alfa / (hr**2 * hz**2)

    # convolve with the gaussian kernel to smooth the solution a bit
    GS_ope_smooth = -F.conv2d(
        GS_ope,
        weight=Gauss_kernel[None, None, ...],
        groups=1,
        padding="same",
    )

    """
    ### this is the scipy implementation
    hr = (RR[0,1,2] - RR[0,1,1]).cpu().detach().numpy()
    hz = (ZZ[0,2,1] - ZZ[0,1,1]).cpu().detach().numpy()
    kr = np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0]))*hz**2
    kz = np.transpose(np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0])))*hr**2

    L_kernel = (kr + kz)/(hr**2*hz**2)

    Df_dr_kernel = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr)

    Lpsi = signal.convolve2d(psi_i, Laplace_kernel, mode='valid')
    # Lpsi = Lpsi/(hr**2*hz**2)
    Dpsi_dr = signal.convolve2d(psi_i, Df_dr_kernel, mode='valid')
    # LHS_conv = Lpsi - Dpsi_dr/RR_in
    GS_ope = Lpsi - Dpsi_dr/RR_in

    ### this is the original tensorflow implementation
    # TF tensors arrives with the shape [batch, H, W, channel]
    pred_tf = pred[:, :, :, None]
    f = tf.expand_dims(tf.constant(pred.detach().cpu().numpy()), axis=-1)
    Laplace_kernel_ds = tf.expand_dims(
        tf.constant(Laplace_kernel.detach().cpu().numpy()), axis=-1
    )
    Df_dr_kernel_ds = tf.expand_dims(
        tf.constant(Df_dr_kernel.detach().cpu().numpy()), axis=-1
    )
    RR_ds = tf.expand_dims(tf.constant(RR.detach().cpu().numpy()), axis=-1)
    ZZ_ds = tf.expand_dims(tf.constant(ZZ.detach().cpu().numpy()), axis=-1)
    

    # f = tf.transpose(f, [3, 1, 2, 0])
    Lpsi = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Laplace_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Lpsi = tf.transpose(
        Lpsi, [3, 1, 2, 0]
    )  # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    Dpsi_dr = tf.nn.depthwise_conv2d(
        f,
        tf.transpose(tf.expand_dims(Df_dr_kernel_ds, axis=-1), [1, 2, 0, 3]),
        strides=[1, 1, 1, 1],
        padding="VALID",
    )
    Dpsi_dr = (
        -Dpsi_dr
    )  # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = tf.transpose(Dpsi_dr, [3, 1, 2, 0])
    RR_in = tf.expand_dims(RR_ds[:, 1:-1, 1:-1], axis=-1)
    Dpsi_dr = tf.math.divide(Dpsi_dr, RR_in)

    GS_ope = Lpsi - Dpsi_dr

    hr = RR_ds[:, 1, 2] - RR_ds[:, 1, 1]
    hz = ZZ_ds[:, 2, 1] - ZZ_ds[:, 1, 1]
    alfa = -2 * (hr**2 + hz**2)

    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr, axis=-1), axis=-1), axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz, axis=-1), axis=-1), axis=-1)
    alfa = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(alfa, axis=-1), axis=-1), axis=-1
    )

    GS_ope = GS_ope * alfa / (hr**2 * hz**2)
    # GS_ope[0,:10,0]

    GS_ope = tf.nn.conv2d(GS_ope, Gauss_tensor, strides=[1, 1, 1, 1], padding="SAME")
    GS_ope = tf.squeeze(GS_ope, axis=-1)
    
    """
    # it appears that there is a - sign missing. Check out why!
    return -GS_ope_smooth.squeeze()


class PDELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.register_buffer(
            "Gauss_kernel", torch.tensor(Gauss_kernel, dtype=torch.float32)
        )

    def forward(
        self,
        pred: Tensor,
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        rhs_computed = _compute_grad_shafranov_operator(
            pred, Laplace_kernel, Df_dr_kernel, RR, ZZ, self.Gauss_kernel
        )
        return self.mse(rhs_computed, rhs)


class PlaNetLoss(nn.Module):
    def __init__(self, scale_mse: float = 1.0, scale_pde: float = 1.0):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_pde = PDELoss()
        self.scale_mse = scale_mse
        self.scale_pde = scale_pde

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        pde_loss = self.scale_pde * self.loss_pde(
            pred=pred,
            rhs=rhs,
            Laplace_kernel=Laplace_kernel,
            Df_dr_kernel=Df_dr_kernel,
            RR=RR,
            ZZ=ZZ,
        )
        return mse_loss + pde_loss
