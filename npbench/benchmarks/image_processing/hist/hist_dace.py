import numpy as np
import dace

@dace.program
def hist(image: dace.uint8[2560, 1536, 3], output: dace.uint8[2560, 1536, 3]):
    imagef = image.astype(np.float32)

    gray = np.empty((imagef.shape[0], imagef.shape[1]), dtype=np.float32)
    gray[:,:] = 0.299 * imagef[:,:,0] + 0.587 * imagef[:,:,1] + 0.114 * imagef[:,:,2]

    Cr = (imagef[:,:,0] - gray) * 0.713 + 128
    Cb = (imagef[:,:,2] - gray) * 0.564 + 128

    gray[gray < 0] = 0
    gray[gray > 255] = 255
    binned_gray = gray.astype(np.uint8)

    # histogram
    hist = np.ndarray((256,), dtype=np.float32)
    for y, x in dace.map[0:binned_gray.shape[0], 0:binned_gray.shape[1]]:
        with dace.tasklet:
            p << binned_gray[y, x]
            out >> hist(1, lambda x, y: x + y)[:]
        
            out[p] = 1

    npixels = gray.shape[0] * gray.shape[1]
    density = hist / npixels

    # cumsum    
    cdf = np.ndarray((256,), dtype=np.float32)
    sum = 0
    for i in range(256):
        sum = sum + density[i]
        cdf[i] = sum

    eq = cdf[binned_gray]
    eq = eq * 255.0
    eq[eq < 0] = 0
    eq[eq > 255] = 255

    red = eq + (Cr - 128) * 1.4
    red[red < 0] = 0
    red[red > 255] = 255

    green = eq - 0.343 * (Cb - 128) - 0.711 * (Cr - 128)
    green[green < 0] = 0
    green[green > 255] = 255

    blue = eq + 1.765 * (Cb - 128)
    blue[blue < 0] = 0
    blue[blue > 255] = 255

    output[:,:,0] = red.astype(np.uint8)
    output[:,:,1] = green.astype(np.uint8)
    output[:,:,2] = blue.astype(np.uint8)

    return output
