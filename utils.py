from skimage.transform import radon, iradon
import numpy as np
from scipy.sparse import csr_matrix, linalg, save_npz, load_npz, vstack
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from PIL import Image


def load_data(file):
    """Load data from .MAT file"""
    return np.array(sio.loadmat(file)['imgs'])

def sparse(A):
    return csr_matrix(A)

def radon_matrix(img_shape):
    """Return radon matrix forward operator"""
    theta = np.array([t for t in range(1, 181, 2)])
    sampler = lambda x: radon(np.reshape(x, img_shape), theta=theta)
    A = get_matrix_from_operator(sampler, np.prod(img_shape))
    A = sparse(A)
    return A

def FBP(sinogram, theta):
    return iradon(sinogram, theta, filter_name='ramp')

def get_matrix_from_operator(Aop, n_pixels):
    """Return operator matrix given""" 
    z = np.zeros(n_pixels)
    for i in range(n_pixels):
        print(f'pix = {i}/{n_pixels}')
        z *= 0
        z[i] = 1
        s = Aop(z)
        if i == 0:
            A = np.zeros((np.prod(s.shape), n_pixels))
        A[:, i] = s.flatten()
    return A

def get_M(sin_shape, undersampling_method, undersampling_rate):
    """Return undersampled matrix M in vector format for 
    elementwise multiplication.
    
    Inputs:
    sin_shape:               sinogram size tuple (r, theta)
    undersampling_method:    either 'limited_angle' or 'sparse_view'
    undersampling_rate:      percentage of undersampling
    
    Output:
    Undersampling vector M for elementwise multiplication
    """
    M = np.zeros(sin_shape)
    if undersampling_rate == 0:
        M = M + 1
    elif undersampling_rate <= 1:
        if undersampling_method.lower() == "limited_angle":
            n_idx = undersampling_rate * sin_shape[1]
            idx = int(n_idx / 2)
            M[:, idx : -idx] = 1

        elif undersampling_method.lower() == "sparse_view":
            n_idx = (1 - undersampling_rate) * sin_shape[1]
            M[:, 0:sin_shape[1]:int(sin_shape[1]/n_idx)] = 1

        else:
            raise ValueError('undersampling_method not found. Accepted values are: ["limited_angle", "sparse_view"]')
    else:
        raise ValueError('undersampling_rate not valid. Accepted values are in the range: [0, 1]')

    M = M.reshape(-1)
    return M

def l21_norm(V):
    """Calculate l_(2,1) matrix norm"""
    return np.sum(np.sqrt(np.sum(V.reshape((-1, V.shape[-1]))**2, axis=1)), axis=0)

def get_data(file, sigma=1):
    """Forward simulate image to sinogram"""
    x = load_data(file)
    Ax = radon_matrix(d.shape) @ d 
    s = Ax + np.random.normal(loc=0, scale=sigma, size=Ad.shape)
    s = M(s, undersampling_method='sparse_view', undersampling_rate=0.5) # s = M(Ax + noise)
    return s, x # noisy sinogram to be reconstructed, and ground truth x


def generate_cyc_difference_matrix_for_image(sz, *args):
    """Generate derivative as sparse matrix"""
    P = np.prod(sz)

    ndif = 1
    spacing = [1, 1, 1]
    
    if len(args) > 0:
        ndif = args[0]
        
    if len(args) > 1:
        spacing = args[1]
    
    if ndif == 1:
        if len(sz) == 1:
            s1 = sz[0]
     
            K = np.prod(sz)
            m1 = np.arange(1, np.prod(sz) + 1)
            m2 = np.roll(m1, -1)
            A1 = csr_matrix((np.ones(K), (m1-1, m2-1)), shape=(K, P)) - csr_matrix((np.ones(K), (m1-1, m1-1)), shape=(K, P))
            A1 /= spacing[0]
        
            A = A1
            
        elif len(sz) == 2:
            s1, s2 = sz
            
            K = np.prod(sz)
            m1 = np.arange(1, np.prod(sz) + 1)
            m2 = np.roll(m1, -1)
            A1 = csr_matrix((np.ones(K), (m1-1, m2-1)), shape=(K, P)) - csr_matrix((np.ones(K), (m1-1, m1-1)), shape=(K, P))
            A1 /= spacing[0]
        
            m1, m2 = np.meshgrid(np.arange(1, s1+1), np.arange(1, s2+1))
            mm1, mm2 = np.meshgrid(np.arange(1, s1+1), np.roll(np.arange(1, s2+1), -1))
            tm = (m1 + (m2 - 1) * s1).flatten()
            tp = (mm1 + (mm2 - 1) * s1).flatten()
            K = tp.size
            A2 = csr_matrix((np.ones(K), (tp-1, tm-1)), shape=(K, P)) - csr_matrix((np.ones(K), (tm-1, tm-1)), shape=(K, P))
            A2 /= spacing[1]
        
            A = vstack((A1, A2))

    return A    
        
if __name__ == '__main__':
    print(generate_cyc_difference_matrix_for_image((5,5)).toarray())


    from PIL import Image
    file = '../../invprob/Exercises/projects/2d_knee.mat'
    d = np.asarray(Image.fromarray(load_data(file)[:,:,1]).resize((64,64))) # downsample for faster debugging
    print(d.shape)

    # save_npz('radon_mat.npz', radon_matrix(d.shape))
    A = load_npz('radon_mat.npz')
    # plt.imshow(A.todense())
    # plt.show()
    s = A @ d.flatten()

    # Derivatives
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
    D = generate_cyc_difference_matrix_for_image((64,64))
    D1 = D[:64*64,:]
    D2 = D[64*64:,:]
    DD1x = D1.transpose() @ (D1 @ d.flatten())
    DD2x = D2.transpose() @ (D2 @ d.flatten())
    axs[0].imshow(d, cmap='gray')
    axs[1].imshow(np.reshape(DD1x, (64,64)), cmap='gray')
    axs[2].imshow(np.reshape(DD2x, (64,64)), cmap='gray')
    plt.show()
    
    # Undersampling
    fig, axs = plt.subplots(1,3)
    sin_shape = (64,90)
    M1 = M(sin_shape, undersampling_method='limited_angle', undersampling_rate=0.7)
    M2 = M(sin_shape, undersampling_method='sparse_view', undersampling_rate=0.7)
    print(M1.shape, s.shape)
    axs[0].imshow(np.reshape(s, sin_shape))
    axs[1].imshow(np.reshape(M1*s, sin_shape))
    axs[2].imshow(np.reshape(M2*s, sin_shape))
    plt.show()


    fig, axs = plt.subplots(1,3)
    for i,m in enumerate([M1, M2]):
        inv = A.transpose() @ A.multiply(m.reshape((-1,1)))
        x, err = cg(inv, A.transpose() @ (m * s), maxiter=300) # At(Ax - s) => x = (AtMt*MA)^-1(AtMt*s)
        axs[i].imshow(np.reshape(x, (64,64)), cmap='gray')
        axs[i].set_title('limited_angle' if i == 0 else 'sparse_view')
    axs[2].imshow(d, cmap='gray')
    plt.show()