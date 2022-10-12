from skimage import measure,io
import os
from PIL import Image
import numpy as np
import csv
from skimage.util.arraycrop import crop

def irregularImagePSNR(im_true, im_test, mask, data_range=255):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

     Parameters
     ----------
     im_true : ndarray
         Ground-truth image.
     im_test : ndarray
         Test image.
     data_range : int 255
         The data range of the input image
     mask : (0,1) Binary ndarray
     Returns
     -------
     psnr : float
         The PSNR metric.
    reference: skimage.measure.compare_psnr(), skimage.measure.compare_mse()
    """
    img_true = im_true * mask
    img_test = im_test * mask
    total_numb = mask.size
    #print(total_numb)
    nonzero_numb = np.count_nonzero(mask)
    #print(nonzero_numb)
    tempmes = measure.compare_mse(img_true, img_test)
    mse = tempmes * total_numb / nonzero_numb
    psnr = 10 * np.log10((data_range ** 2) / mse)

    return psnr


def irregularImageSSIM(img1, img2, mask, data_range=255, gaussian_weights=True, full=True):
    """ Compute the peak signal to noise ratio (PSNR) for an image.

     Parameters
     ----------
     img1,img2 : ndarray

     data_range : int
         The data range of the input image
     mask : (0,1) Binary ndarray
     Returns
     -------
     ssim : float
          The mean structural similarity over the image* mask
    reference: skimage.measure.compare_ssim()
    """

    _, ssim_array =measure.compare_ssim(img1, img2, data_range=data_range, gaussian_weights=gaussian_weights, full=full)
    valid_ssim = ssim_array * mask
    nonzero_numb = np.count_nonzero(mask)
    ssim = np.sum(valid_ssim) / nonzero_numb

    return ssim


'''
def imagePSNR(im_true, im_test, data_range=255):

    return measure.compare_psnr(im_true=im_true, im_test=im_test, data_range=data_range)


def imageSSIM(img1, img2, data_range=255, gaussian_weights=True, full=True):
    """ Compute the peak signal to noise ratio (PSNR) for an image.
    compared with skimage.measure.compare_ssim(), this function  doesnot avoid edge effects
     (mssim = crop(S, pad).mean())

     Parameters
     ----------
     img1,img2 : ndarray

     data_range : int
         The data range of the input image
     Returns
     -------
     ssim : float
          The mean structural similarity over the image* mask
    reference: skimage.measure.compare_ssim()
    """

    _, ssim_array = measure.compare_ssim(img1, img2, data_range=data_range, gaussian_weights=gaussian_weights, full=full)

    mssim = ssim_array.mean()
    return mssim
'''


inImgRDir = './results'
outcsvDir = './results'

#inImgRDir = './testmetrics'
#outcsvDir = './testmetrics'
#imgFolders_List= ['test_latest_long', 'test_latest_short', 'train_latest_long']
imgFolders_List= ['test_200_short']
#imgSize = 512
imgSize = 256

csvHeader=['Filename','HolePSNR','HoleSSIM' ]

for imgfolder in imgFolders_List:
    #print(imgfolder)
    inImgDir = os.path.join(inImgRDir, imgfolder)
    data_type = imgfolder.split('_')[-1]
    csvFname = imgfolder+'_metrics.csv'
    csvFpath = os.path.join(outcsvDir, csvFname)

    if os.path.exists(csvFpath):
        os.remove(csvFpath)

    with open(csvFpath, 'w') as csvfw:
        writer = csv.writer(csvfw)
        writer.writerow(csvHeader)

        total_HolePSNR = 0.0
        total_HoleSSIM = 0.0
        count_image = 0

        imagenames_list = sorted(os.listdir(inImgDir))
        #img_numb = len(imagenames_list)
        for imgname in imagenames_list:
            inImgPath = os.path.join(inImgDir, imgname)
            #imageGroup = Image.open(inImgPath).convert('L')
            imageGroup = Image.open(inImgPath)
            # split imageGroup image into input,  mask,   output, final_output, ground_truth, refer
            #                             real_A, mask_M, fake_B, final_B,      real_B,       refer_B
            # So, each group includes at least 6 images (maybe init_B, fakehole_B and so on, but first six is fixed)
            width, height = imageGroup.size
            imgNumb = width // imgSize
            #print(imgNumb)
            w0 = imgSize

            if imgNumb >= 6: # each group includes at least 6 images
                imgInput = imageGroup.crop((0, 0, w0, height))
                imgMask = imageGroup.crop((w0, 0, w0 * 2, height))
                imgOutput = imageGroup.crop((w0 * 2, 0, w0 * 3, height))
                imgFinal = imageGroup.crop((w0 * 3, 0, w0 * 4, height))
                imgGTruth = imageGroup.crop((w0 * 4, 0, w0 * 5, height))
                imgRefer = imageGroup.crop((w0 * 5, 0, w0 * 6, height))

                csvRow_list = []

                Filename = imgname
                arrayMask = np.array(imgMask)
                validMask = arrayMask // 255
                holeMask = 1 - validMask

                #if np.count_nonzero(holeMask) < 20*20: # skip no hole image group
                #    continue

                img_final = np.array(imgFinal)
                img_gtruth = np.array(imgGTruth)
                img_refer = np.array(imgRefer)

                if data_type == 'short':

                    HolePSNR = irregularImagePSNR(img_gtruth, img_final, holeMask, data_range=255)
                    HoleSSIM = irregularImageSSIM(img_gtruth, img_final, holeMask, data_range=255, gaussian_weights=True, full=True)

                    count_image = count_image + 1
                    print('short', count_image)

                if data_type == 'long': #  no ground truth, so use img_refer to repalce ground truth

                    HolePSNR = irregularImagePSNR(img_refer, img_final, holeMask, data_range=255)
                    HoleSSIM = irregularImageSSIM(img_refer, img_final, holeMask, data_range=255, gaussian_weights=True, full=True)

                    count_image = count_image + 1

                    print('long', count_image)

                csvRow_list = [Filename, HolePSNR, HoleSSIM]

                writer.writerow(csvRow_list)


                total_HolePSNR = total_HolePSNR + HolePSNR
                total_HoleSSIM = total_HoleSSIM + HoleSSIM

            else:
                print('The number of images in this group is Wrong !')


        mean_HolePSNR = total_HolePSNR / count_image
        mean_HoleSSIM = total_HoleSSIM / count_image

        meanRow = ['MeanValue', mean_HolePSNR, mean_HoleSSIM]
        writer.writerow(meanRow)

    print('The %s folder is over for computation of metrics!\n' %(imgfolder))

print('The metrics program is over!')




