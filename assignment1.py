import numpy as np
from PIL import Image
import math
import time
import tqdm
import argparse
from multiprocessing import Process, Array, Manager

def getBlock(plane, blockX, blockY, block_size):
    return plane[block_size*blockY:block_size*blockY + block_size, block_size*blockX:block_size*blockX + block_size]

def readFrames(file_name, height, width, frame_count, save_frames):
    # to be changed manually
    filey4m = file_name + ".y4m"
    fileyuv = file_name + ".yuv"
    if False:
        process = subprocess.Popen(["ffmpeg", "-i", filey4m, fileyuv])
        out = process.communicate()[0]

    f_y = open(fileyuv, "rb")
    frames = np.array([])

    print("Start Reading Frames")
    for k in tqdm.tqdm(range(frame_count)):
        y = f_y.read(width * height)
        y = np.array(list(y)).reshape(height, width)

        u = f_y.read(int(width / 2) * int(height / 2))
        u = np.array(list(u)).reshape(int(height / 2), int(width / 2))

        v = f_y.read(int(width / 2) * int(height / 2))
        v = np.array(list(v)).reshape(int(height / 2), int(width / 2))
        frame = []
        frame.append(y)
        frame.append(u)
        frame.append(v)
        frames = np.append(frames, np.array(frame))
    return frames


def exhaustiveSearchMainLoop(start_index, end_index, frames, height, width, frame_count, block_size, search_window, psnrs, computation_costs):
    matchedFrameY = np.zeros((height, width))
    matchedFrameU = np.zeros((int(height/2), int(width/2)))
    matchedFrameV = np.zeros((int(height/2), int(width/2)))

    for i in tqdm.tqdm(range(start_index, end_index)):
        if i == 0:
            continue
        #reference frame channels
        rfY = frames[3 * i - 3]
        rfU = frames[3 * i - 2]
        rfV = frames[3 * i - 1]

        #current frame channels
        cfY = frames[3 * i]
        cfU = frames[3 * i + 1]
        cfV = frames[3 * i + 2]

        indexingCount = 0
        matrixSubtractionCount = 0
        #iterate over each macro block
        for y in range(int(height/block_size)):
            for x in range(int(width/block_size)):

                macroBlock = getBlock(cfY, x, y, block_size)
                indexingCount += 1

                searchWindowStartX = int((block_size*x)+(block_size/2) - search_window/2)
                searchWindowEndX = int(searchWindowStartX + search_window - block_size)

                searchWindowStartY = int((block_size * y) + (block_size / 2) - search_window/2)
                searchWindowEndY = int(searchWindowStartY + search_window - block_size)

                #check boundaries
                if searchWindowStartX < 0:
                    searchWindowStartX = 0
                elif searchWindowEndX + block_size > width:
                    searchWindowEndX = width - block_size

                if searchWindowStartY < 0:
                    searchWindowStartY = 0
                elif searchWindowEndY + block_size > height:
                    searchWindowEndY = height - block_size

                #print("startX: {}, endX: {}, startY: {}, endY: {}".format(searchWindowStartX, searchWindowEndX, searchWindowStartY, searchWindowEndY))

                referenceBlockX = 0
                referenceBlockY = 0
                minError = 210840234
                for j in range(searchWindowStartY, searchWindowEndY):
                     for k in range(searchWindowStartX, searchWindowEndX):
                        referenceBlock = rfY[j:j+block_size, k:k+block_size]
                        indexingCount +=1
                        mse = (np.square(macroBlock - referenceBlock)).mean(axis = None)
                        matrixSubtractionCount += 1
                        if minError > mse:
                            minError = mse
                            referenceBlockX = k
                            referenceBlockY = j

                matchedFrameY[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size] = rfY[referenceBlockY:referenceBlockY+block_size, referenceBlockX:referenceBlockX+block_size]
                matchedFrameU[y*int(block_size/2):(y+1)*int(block_size/2), x*int(block_size/2):(x+1)*int(block_size/2)] = rfU[int(referenceBlockY/2):int(referenceBlockY/2)+int(block_size/2), int(referenceBlockX/2):int(referenceBlockX/2)+int(block_size/2)]
                matchedFrameV[y * int(block_size / 2):(y + 1) * int(block_size / 2), x * int(block_size / 2):(x + 1) * int(block_size / 2)] = rfV[int(referenceBlockY/2):int(referenceBlockY/2)+int(block_size/2), int(referenceBlockX/2):int(referenceBlockX/2)+int(block_size/2)]
                indexingCount += 6

        difY = np.abs(cfY - matchedFrameY)
        difU = np.abs(cfU - matchedFrameU)
        difV = np.abs(cfV - matchedFrameV)

        residualImage = yuvImage(difY, difU, difV, width, height)
        estimatedImage = yuvImage(matchedFrameY, matchedFrameU, matchedFrameV, width, height)

        mse = (np.sum(np.square(cfY - matchedFrameY))+np.sum(np.square(cfU - matchedFrameU)) + np.sum(np.square(cfV - matchedFrameV)))/(height*width + 2*(height/2*width/2))
        psnr = 20*math.log(255, 10) - 10*math.log(mse, 10)
        psnrs.append((i, psnr))
        computation_costs.append((i, indexingCount, matrixSubtractionCount))
        residualImage.save("exhaustive/residual/" + str(i) +".png")
        estimatedImage.save("exhaustive/estimated/" + str(i) + ".png")

def exhaustiveSearch(frames, height, width, frame_count, block_size, search_window, process_count = 1):

    processs = []
    framePerProcess = frame_count/process_count
    psnrs = Manager().list()
    computationCosts = Manager().list()
    if process_count == 1:
        exhaustiveSearchMainLoop(0, frame_count, frames, height, width, frame_count, block_size, search_window, psnrs, computationCosts)
    else :
        for processNo in range(process_count):
            t = Process(target=exhaustiveSearchMainLoop, args=(processNo*int(framePerProcess), (processNo + 1)*int(framePerProcess), frames, height, width, frame_count, block_size, search_window, psnrs, computationCosts))
            processs.append(t)
            t.start()

        for processNo in range(len(processs)):
            processs[processNo].join()

    psnrSum = 0
    indexingSum = 0
    matrixSubtractionSum = 0
    for i in range(len(psnrs)):
        psnrSum += psnrs[i][1]
        indexingSum += computationCosts[i][1]
        matrixSubtractionSum += computationCosts[i][2]

    print("Exhaustive Search Results:")
    print("Average psnr: {}".format(psnrSum / len(psnrs)))
    print("Average number of indexing: {} \nAverage number of matrix subtraction: {}".format(
        indexingSum / len(computationCosts), matrixSubtractionSum / len(computationCosts)))


def diamondSearchMainLoop(start_index, end_index, matchedFrameY, matchedFrameU, matchedFrameV, frames, height, width,
                          frame_count, block_size, search_window, psnrs, computational_costs):

    def diamondSearchPattern(ldsp=True, searchX=0, searchY=0, S=2, indexing_count=0, matrix_subtraction_count = 0):

        minmse = 685748648
        minmseWindowOffsetY = 0
        minmseWindowOffsetX = 0

        for a in range(-S, S + 1):
            for b in range(-S, S + 1):
                if (searchY + a >= searchWindowStartY and searchY + a <= searchWindowEndY and searchX + b >= searchWindowStartX and searchX + b <= searchWindowEndX) \
                        and (((a == 0 and b == 0) or abs(a) + abs(b) == S) and (
                        0 <= searchX + b and width - block_size >= b + searchX and 0 <= a + searchY and height - block_size >= a + searchY)):
                    candidateReferenceBlock = rfY[searchY + a: searchY + a + block_size,
                                              searchX + b: searchX + b + block_size]
                    indexing_count += 1
                    mse = (np.square(macroBlockY - candidateReferenceBlock)).mean(axis=None)
                    matrix_subtraction_count += 1
                    if minmse > mse:
                        minmse = mse
                        minmseWindowOffsetX = b
                        minmseWindowOffsetY = a
                else:
                    continue
        return minmse, minmseWindowOffsetX, minmseWindowOffsetY, indexing_count, matrix_subtraction_count

    for i in tqdm.tqdm(range(start_index, end_index)):
        if i == 0:
            continue
        # reference frame channels
        rfY = frames[3 * i - 3]
        rfU = frames[3 * i - 2]
        rfV = frames[3 * i - 1]

        # current frame channels
        cfY = frames[3 * i]
        cfU = frames[3 * i + 1]
        cfV = frames[3 * i + 2]

        indexingCount = 0
        matrixSubtractionCount = 0

        for y in range(int(height / block_size)):
            for x in range(int(width / block_size)):
                macroBlockY = getBlock(cfY, x, y, block_size)
                indexingCount += 1

                searchWindowStartX = int((block_size * x) + (block_size / 2) - search_window / 2)
                searchWindowEndX = int(searchWindowStartX + search_window - block_size)

                searchWindowStartY = int((block_size * y) + (block_size / 2) - search_window / 2)
                searchWindowEndY = int(searchWindowStartY + search_window - block_size)

                # check boundaries
                if searchWindowStartX < 0:
                    searchWindowStartX = 0
                elif searchWindowEndX + block_size > width:
                    searchWindowEndX = width - block_size

                if searchWindowStartY < 0:
                    searchWindowStartY = 0
                elif searchWindowEndY + block_size > height:
                    searchWindowEndY = height - block_size

                macroBlockStartX = x * block_size
                macroBlockStartY = y * block_size

                minmse, windowOffsetX, windowOffsetY, indexingCount, matrixSubtractionCount = diamondSearchPattern(ldsp=True, searchX=macroBlockStartX,
                                                                            searchY=macroBlockStartY, indexing_count=indexingCount, matrix_subtraction_count=matrixSubtractionCount)
                totalOffsetX = windowOffsetX
                totalOffsetY = windowOffsetY

                while True:
                    if windowOffsetX == 0 and windowOffsetY == 0:
                        break
                    minmse, windowOffsetX, windowOffsetY, indexingCount, matrixSubtractionCount = diamondSearchPattern(ldsp=True,
                                                                                searchX=macroBlockStartX + totalOffsetX,
                                                                                searchY=macroBlockStartY + totalOffsetY, indexing_count=indexingCount, matrix_subtraction_count=matrixSubtractionCount)
                    totalOffsetX += windowOffsetX
                    totalOffsetY += windowOffsetY

                minmse, windowOffsetX, windowOffsetY, indexingCount, matrixSubtractionCount = diamondSearchPattern(ldsp=False,
                                                                            searchX=macroBlockStartX + totalOffsetX,
                                                                            searchY=macroBlockStartY + totalOffsetY, indexing_count=indexingCount, matrix_subtraction_count=matrixSubtractionCount)

                matchedFrameY[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = rfY[
                                                                                                          macroBlockStartY + totalOffsetY: macroBlockStartY + totalOffsetY + block_size,
                                                                                                          totalOffsetX + macroBlockStartX: macroBlockStartX + totalOffsetX + block_size]
                matchedFrameU[y * int(block_size / 2):(y + 1) * int(block_size / 2),
                x * int(block_size / 2):(x + 1) * int(block_size / 2)] = rfU[
                                                                         int(macroBlockStartY / 2) + int(
                                                                             totalOffsetY / 2): int(
                                                                             macroBlockStartY / 2) + int(
                                                                             totalOffsetY / 2) + int(block_size / 2),
                                                                         int(totalOffsetX / 2) + int(
                                                                             macroBlockStartX / 2): int(
                                                                             macroBlockStartX / 2) + int(
                                                                             totalOffsetX / 2) + int(block_size / 2)]

                matchedFrameV[y * int(block_size / 2):(y + 1) * int(block_size / 2),
                x * int(block_size / 2):(x + 1) * int(block_size / 2)] = rfV[
                                                                         int(macroBlockStartY / 2) + int(
                                                                             totalOffsetY / 2): int(
                                                                             macroBlockStartY / 2) + int(
                                                                             totalOffsetY / 2) + int(block_size / 2),
                                                                         int(totalOffsetX / 2) + int(
                                                                             macroBlockStartX / 2): int(
                                                                             macroBlockStartX / 2) + int(
                                                                             totalOffsetX / 2) + int(block_size / 2)]
                indexingCount += 6

        difY = np.abs(cfY - matchedFrameY)
        difU = np.abs(cfU - matchedFrameU)
        difV = np.abs(cfV - matchedFrameV)

        mse = (np.sum(np.square(cfY - matchedFrameY))+np.sum(np.square(cfU - matchedFrameU)) + np.sum(np.square(cfV - matchedFrameV)))/(height*width + 2*(height/2*width/2))
        psnr = 20*math.log(255, 10) - 10*math.log(mse, 10)
        psnrs.append((i, psnr))
        computational_costs.append((i, indexingCount, matrixSubtractionCount))
        residualImage = yuvImage(difY, difU, difV, width, height)
        estimatedImage = yuvImage(matchedFrameY, matchedFrameU, matchedFrameV, width, height)

        residualImage.save("diamond/residual/" + str(i) + ".png")
        estimatedImage.save("diamond/estimated/" + str(i) + ".png")


def diamondSearch(frames, height, width, frame_count, block_size, search_window, process_count=1):
    matchedFrameY = np.zeros((height, width))
    matchedFrameU = np.zeros((int(height / 2), int(width / 2)))
    matchedFrameV = np.zeros((int(height / 2), int(width / 2)))

    psnrs = Manager().list()
    computationCosts = Manager().list()

    if process_count == 1:
        diamondSearchMainLoop(0, frame_count, matchedFrameY, matchedFrameU, matchedFrameV, frames, height, width, frame_count, block_size, search_window, psnrs, computationCosts)
    else:
        processs = []
        framePerProcess = frame_count/process_count

        for processNo in range(process_count):
            t = Process(target=diamondSearchMainLoop, args=(processNo*int(framePerProcess), (processNo + 1)*int(framePerProcess), matchedFrameY, matchedFrameU, matchedFrameV, frames, height, width, frame_count, block_size, search_window, psnrs, computationCosts))
            processs.append(t)
            t.start()

        for processNo in range(len(processs)):
            processs[processNo].join()
    psnrSum = 0
    indexingSum = 0
    matrixSubtractionSum = 0
    for i in range(len(psnrs)):
        psnrSum += psnrs[i][1]
        indexingSum += computationCosts[i][1]
        matrixSubtractionSum += computationCosts[i][2]

    print("Diamond Search Result:")
    print("Average psnr: {}".format(psnrSum/len(psnrs)))
    print("Average number of indexing: {} \nAverage number of matrix subtraction: {}".format(indexingSum/len(computationCosts), matrixSubtractionSum/len(computationCosts)))


#convert YUV image to RGB
def yuvImage(y, u, v, width, height):
    image_out = Image.new("RGB", (width, height))
    pix = image_out.load()
    for m in range(0, height):
        for n in range(0, width):
            Y_val = y[m, n]
            U_val = u[int(m / 2), int(n / 2)]
            V_val = v[int(m / 2), int(n / 2)]

            B = 1.164 * (Y_val - 16) + 2.018 * (U_val - 128)
            G = 1.164 * (Y_val - 16) - 0.813 * (V_val - 128) - 0.391 * (U_val - 128)
            R = 1.164 * (Y_val - 16) + 1.596 * (V_val - 128)
            pix[n, m] = int(R), int(G), int(B)

    return image_out

#method to run algorithms from
def run(file_name, height, width, block_size, search_window, frame_count, save_frames, process_count):

    frames = readFrames(file_name, height, width, frame_count, save_frames)
    exhaustiveSearch(frames, height, width, frame_count, block_size, search_window, process_count,)
    diamondSearch(frames, height, width, frame_count, block_size, search_window, process_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="The name of the video file in .y4m format")
    parser.add_argument("height", help="Height of the video frames", type=int)
    parser.add_argument("width", help="Width of the video frames", type=int)
    parser.add_argument("frame", help="Number of frames", type=int)
    parser.add_argument("search_window", help="Size of the search window", type=int, default=50)
    parser.add_argument("-b", "--block_size", help="The macroblock size", type=int, default=16)
    parser.add_argument("-s", "--save_frames", help="Whether to save the frames or not", action="store_true")
    parser.add_argument("-p", "--process_count", help="Number of processes to run the algorithms with(deafult = 1)", type=int, default=1)
    args = parser.parse_args()


    run(args.file, args.height, args.width, args.block_size, args.search_window, args.frame, args.save_frames, args.process_count)

if __name__ == "__main__":
    main()