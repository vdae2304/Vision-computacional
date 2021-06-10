import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def filter2d(img, kernel, padding = None):
    '''
    Aplica un filtro (convolución) sobre una imagen.
    Argumentos:
        img : ndarray de tamaño (height, width, channels)
            Imagen sobre la cual se aplicará el filtro.
        kernel : ndarray de tamaño (kheight, kwidth)
            Kernel a utilizar.
        padding : {None, 'constant', 'edge', 'reflect', 'symmetric', 'wrap'}
            Relleno a agregar en los bordes. Por defecto None.
    Devuelve:
        img_out : ndarray de tamaño (height, width, channels)
            Imagen resultante.
    '''
    kheight, kwidth = kernel.shape
    vpad = kheight // 2
    hpad = kwidth // 2
    
    if kheight % 2 == 0 or kwidth % 2 == 0:
        raise ValueError('El tamaño del kernel debe ser impar.')
    if padding is not None:
        img = np.pad(img, [(vpad, vpad), (hpad, hpad), (0, 0)], padding)
    
    height, width, channels = img.shape
    img_out = img.astype(float)
    img_out[vpad:(height - vpad), hpad:(width - hpad), :] = 0.0
    
    for i in range(kheight):
        for j in range(kwidth):
            img_out[vpad:(height - vpad), hpad:(width - hpad), :] += \
                kernel[i, j] * img[i:(height - kheight + i + 1), 
                                   j:(width - kwidth + j + 1), :]
    
    if padding is not None:
        img_out = img_out[vpad:(height - vpad), hpad:(width - hpad), :]
    img_out = img_out.astype(img.dtype)
    
    return img_out

def gaussian_kernel(ksize, std = 1):
    '''
    Devuelve un kernel gaussiano sobre una dimensión.
    Argumentos:
        ksize : int
            Tamaño del kernel.
        std : float
            Desviación estándar.
    Devuelve:
        kernel : ndarray de tamaño (ksize,)
            Kernel correspondiente.
    '''
    if ksize % 2 == 0:
        raise ValueError('El tamaño del kernel debe ser impar.')
    kernel = stats.norm.pdf(np.arange(ksize) - ksize//2, 0, std)
    kernel /= np.sum(kernel)   
    return kernel

def gaussian_blur(img, ksize, std = 1, padding = None):
    '''
    Aplica un desenfoque gaussiano sobre una imagen.
    Argumentos:
        img : ndarray de tamaño (height, width, channels)
            Imagen sobre la cual se aplicará el filtro.
        ksize : (int, int)
            Tamaño del kernel.
        std : float
            Desviación estándar.
        padding: {None, 'constant', 'edge', 'reflect', 'symmetric', 'wrap'}
            Relleno a agregar en los bordes.
    Devuelve:
        img_out : ndarray de tamaño (height, width, channels)
            Imagen resultante.
    '''
    kheight, kwidth = ksize
    vkernel = gaussian_kernel(kheight, std).reshape(-1, 1)
    hkernel = gaussian_kernel(kwidth, std).reshape(1, -1)
    
    img_out = img.copy()
    img_out = filter2d(img_out, hkernel, padding)
    img_out = filter2d(img_out, vkernel, padding)
    
    return img_out

def median_filter(img, ksize, padding = None):
    '''
    Aplica un filtro de mediana sobre una imagen.
    Argumentos:
        img : ndarray de tamaño (height, width, channels)
            Imagen sobre la cual se aplicará el filtro.
        ksize : (int, int)
            Tamaño del kernel.
        padding : {None, 'constant', 'edge', 'reflect', 'symmetric', 'wrap'}
            Relleno a agregar en los bordes. Por defecto None.
    Devuelve:
        img_out : ndarray de tamaño (height, width, channels)
            Imagen resultante.
    '''
    kheight, kwidth = ksize
    vpad = kheight // 2
    hpad = kwidth // 2
    
    if kheight % 2 == 0 or kwidth % 2 == 0:
        raise ValueError('El tamaño del kernel debe ser impar.')
    if padding is not None:
        img = np.pad(img, [(vpad, vpad), (hpad, hpad), (0, 0)], padding)
    
    height, width, nchannels = img.shape
    img_out = img.copy()
    
    for i in range(vpad, height - vpad):
        for j in range(hpad, width - hpad):
            window = img[(i - vpad):(i + vpad + 1), 
                         (j - hpad):(j + hpad + 1), :]
            img_out[i, j, :] = np.median(window, axis = (0, 1))
    
    if padding is not None:
        img_out = img_out[vpad:(height - vpad), hpad:(width - hpad), :]
    
    return img_out

def sobel(img, dx, dy, padding = None):
    '''
    Calcula las derivadas parciales de la imagen utilizando el operador de 
    Sobel.
    Argumentos:
        img : ndarray de tamaño (height, width, channels)
            Imagen sobre la cual se calcularán las derivadas.
        dx : int
            Orden de la derivada sobre el eje horizontal.
        dy : int
            Orden de la derivada sobre el eje vertical.
        padding : {None, 'constant', 'edge', 'reflect', 'symmetric', 'wrap'}
            Relleno a agregar en los bordes. Por defecto None.
    Devuelve:
        img_out : ndarray de tamaño (height, width, channels)
            Imagen resultante.
    '''
    img_out = img.copy()
    
    for i in range(dx):
        kernel1 = np.array([[1, 0, -1]])
        kernel2 = np.array([[1], [2], [1]])
        img_out = filter2d(img_out, kernel1, padding)
        img_out = filter2d(img_out, kernel2, padding)
        
    for i in range(dy):
        kernel1 = np.array([[1, 2, 1]])
        kernel2 = np.array([[1], [0], [-1]])
        img_out = filter2d(img_out, kernel1, padding)
        img_out = filter2d(img_out, kernel2, padding)
    
    return img_out

def anisotropic_diffusion(img, coef, K, step = 0.1, niter = 10):
    '''
    Aplica el algoritmo de difusión anisotrópica sobre una imagen.
    Argumentos:
        img : ndarray de tamaño (height, width, channels)
            Imagen sobre la cual se aplicará el algoritmo.
        coef : {'exp', 'inv'}
            Coeficiente de difusión a utilizar. Debe ser alguno de los 
            siguientes:
                'exp':
                    exp(-(|grad|/K)**2)
                'inv':
                    1/(1 + (|grad|/K)**2)
        K : float
            Grado de sensibilidad a los bordes.
        step : float
            Tamaño de paso entre cada iteración.    
        niter : int
            Número de iteraciones a realizar.
    Devuelve:
        img_out : ndarray de tamaño (height, width, channels)
            Imagen resultante.
    '''
    img_out = img.astype(float)
    
    for t in range(niter):
        grad_x = sobel(img_out, 1, 0, 'reflect')
        grad_y = sobel(img_out, 0, 1, 'reflect')
        grad_squared_norm = grad_x**2 + grad_y**2
        
        if coef == 'exp':
            c = np.exp(-(grad_squared_norm / K**2))
        elif coef == 'inv':
            c = 1 / (1 + grad_squared_norm / K**2)
        else:
            raise ValueError('coef debe ser alguno de exp o inv.')
        
        div_x = sobel(c*grad_x, 1, 0, 'reflect')
        div_y = sobel(c*grad_y, 0, 1, 'reflect') 
        img_out += step * (div_x + div_y)
    
    img_out = img_out.astype(img.dtype)
    
    return img_out

def plot_images(nrows, ncols, images, fig = None, title = [], cmap = []):
    '''
    Grafica varias imágenes sobre una misma figura.
    Argumentos
        nrows : int
            Número de renglones.
        ncols : int
            Número de columnas.
        images : list
            Lista de imágenes.
        fig : matplotlib.figure.Figure
            Figura sobre la cual graficar.
        title : list, opcional
            Títulos de cada subgrafica.
        cmap : list, opcional
            Mapa de colores de cada subgráfica.
    '''
    if fig is None:
        fig = plt.figure()
        
    for i, img in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.axis('off')
        if len(title) > i:
            ax.set_title(title[i])
        if len(cmap) > i:
            ax.imshow(img, cmap = cmap[i])
        else:
            ax.imshow(img)
    
    return fig