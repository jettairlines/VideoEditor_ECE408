// ECE 408 Final Project - Video Processing Acceleration
/*
* Notes:
* - Uses software pipelining with two Cuda Streams for data transfer and kernel
* execution concurrently. Interleaved processing of two frames.
* - Frames are organized as two dimensional uchar arrays (RGB values) frame[i][j]
* where i = video height and j = video width * 3 for each color.
* - Prefetching is used in loops, by default fetches a cache line four iterations away
* - wsize controls the underlying vertex array of the corner detection kernel
*/

// CUDA 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
// STDLIB
#include <fstream>
#include <iostream>
#include <math.h>
#include <assert.h>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\ml.h>
#include <opencv\cxcore.h>

#define uchar unsigned char
#define wsize 4

using namespace cv;
using namespace std;

cudaError_t kernelCall(uchar*** frame_p, int count, int w, int h, int func);
cudaError_t kernelCall_Conv(uchar*** frame_p, int count, int w, int h, int func);
cudaError_t kernelCall_Edge(uchar*** frame_p, int count, int w, int h, int thresMin, int thresMax);
cudaError_t kernelCall_UpScale(uchar** frame_p, uchar** frame_l, int w, int h, int nw, int nh);
cudaError_t kernelCall_Detect(uchar*** frame_p, int count, int w, int h, int thresMin, int thresMax, int func);

int cudaWrapper(uchar*** frame_p, int count, int w, int h, int func, int thresMin, int thresMax, int nw, int nh);
int upScale(int nw, int nh);
int processImage(int thresMin, int thresMax);

void initGaussMask(float* Mask);
void initSobel_X(int* SO);
void initSobel_Y(int* SO);

__constant__ float M[5 * 5];
__constant__ int SO_X[3 * 3];
__constant__ int SO_Y[3 * 3];

__device__ void prefetch(unsigned int addr) {
	asm(" prefetch.global.L1 [ %1 ];": "=r"(addr) : "r"(addr));
}

__global__ void redFilterKernel(uchar *frame, int w, int h)
{
	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * w * 3 + 3 * x;

	// Colors at Pixel Locations
	uchar R, G, B;
	R = frame[pixel];
	G = frame[pixel + 1];
	B = frame[pixel + 2];

	frame[pixel] = 0;
}

__global__ void gaussianBlurKernel(uchar* frame_in, uchar* frame_out, int w, int h) {

	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * w * 3 + 3 * x;

	// Colors at Pixel Locations
	uchar R, G, B;
	R = 0;
	G = 0;
	B = 0;

	int pos_X;
	int pos_Y;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			pos_X = x + i - 2;
			pos_Y = y + j - 2;
			if ((pos_X >= 0) && (pos_X < w)) {
				if ((pos_Y >= 0) && (pos_Y < h)) {
					R += M[i * 5 + j] * frame_in[pos_Y * w * 3 + pos_X * 3];
					G += M[i * 5 + j] * frame_in[pos_Y * w * 3 + pos_X * 3 + 1];
					B += M[i * 5 + j] * frame_in[pos_Y * w * 3 + pos_X * 3 + 2];
				}
			}
		}
	}
	
	// Remap Colors
	frame_out[pixel] = R;
	frame_out[pixel + 1] = G;
	frame_out[pixel + 2] = B;
}

__global__ void sobelOpKernel(uchar* frame, int* gradient, int w, int h) {

	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * w * 3 + 3 * x;

	// Colors at Pixel Locations
	int R, G, B;
	int Gx, Gy;
	Gx = 0;
	Gy = 0;

	// Grayscale Luminosity
	int lum;

	// TODO: Optimization
	int pos_X;
	int pos_Y;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			pos_X = x + i - 1;
			pos_Y = y + j - 1;
			if ((pos_X >= 0) && (pos_X < w)) {
				if ((pos_Y >= 0) && (pos_Y < h)) {

					R = frame[pos_Y * w * 3 + pos_X * 3];
					G = frame[pos_Y * w * 3 + pos_X * 3 + 1];
					B = frame[pos_Y * w * 3 + pos_X * 3 + 2];
					
					lum = (int) (0.21 * R + 0.72 * G + 0.07 * B);
					Gx += SO_X[i * 3 + j] * lum;
					Gy += SO_Y[i * 3 + j] * lum;

				}
			}
		}
	}

	float mag = Gx * Gx + Gy * Gy;
	int magnitude = (int) sqrt(mag);
	gradient[y * w + x] = magnitude;
}

__global__ void edgeHystKernel(uchar* frame, int* gradient, int w, int h, int thresMin, int thresMax) {

	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * w * 3 + 3 * x;
	int mag = gradient[y * w + x];

	frame[pixel] = 0;
	frame[pixel + 1] = 0;
	frame[pixel + 2] = 0;
	
	if (mag > thresMax) {
		frame[pixel] = 255;
		frame[pixel + 1] = 255;
		frame[pixel + 2] = 255;
	}
	else if (mag < thresMax && mag > thresMin) {
		if (x + 1 < w && x - 1 >= 0 && y + 1 < h && y - 1 >= 0) {
			if (gradient[y * w + x + 1] > thresMax) {
				frame[pixel] = 255;
				frame[pixel + 1] = 255;
				frame[pixel + 2] = 255;
			}
			if (gradient[y * w + x - 1] > thresMax) {
				frame[pixel] = 255;
				frame[pixel + 1] = 255;
				frame[pixel + 2] = 255;
			}
			if (gradient[(y + 1) * w + x] > thresMax) {
				frame[pixel] = 255;
				frame[pixel + 1] = 255;
				frame[pixel + 2] = 255;
			}
			if (gradient[(y - 1) * w + x] > thresMax) {
				frame[pixel] = 255;
				frame[pixel + 1] = 255;
				frame[pixel + 2] = 255;
			}
		}
	}
	
}

__global__ void sobelOpAngleKernel(uchar* frame, int* gradient, int* angle, float* Ix, float* Iy, int w, int h) {

	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * w * 3 + 3 * x;

	// Colors at Pixel Locations
	int R, G, B;
	int Gx, Gy;
	Gx = 0;
	Gy = 0;

	// Grayscale Luminosity
	int lum;

	// TODO: Optimization
	int pos_X;
	int pos_Y;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			pos_X = x + i - 1;
			pos_Y = y + j - 1;
			if ((pos_X >= 0) && (pos_X < w)) {
				if ((pos_Y >= 0) && (pos_Y < h)) {

					R = frame[pos_Y * w * 3 + pos_X * 3];
					G = frame[pos_Y * w * 3 + pos_X * 3 + 1];
					B = frame[pos_Y * w * 3 + pos_X * 3 + 2];

					lum = (int)(0.21 * R + 0.72 * G + 0.07 * B);
					Gx += SO_X[i * 3 + j] * lum;
					Gy += SO_Y[i * 3 + j] * lum;

				}
			}
		}
	}
	
	int ang = (int) (100.0 * atan(((float)Gy) / ((float)Gx)));
	float mag = Gx * Gx + Gy * Gy;
	int magnitude = (int)sqrt(mag);
	gradient[y * w + x] = magnitude;
	angle[y * w + x] = ang;
	Ix[y * w + x] = (float) Gx / 32.0;
	Iy[y * w + x] = (float) Gy / 32.0;
	
}

__global__ void edgeVertexKernel(uchar* frame, int* gradient, uchar* vertices, int* angle, float* Ix, float* Iy,  int hw, int hh, int thresMin, int thresMax) {

	// Index Variables
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int nx = bx * blockDim.x + tx;
	int ny = by * blockDim.y + ty;
	int x = (int) (nx * wsize);
	int y = (int) (ny * wsize);
	int w = hw * wsize;
	int h = hh * wsize;
	int vertex = 0;
	int mag = 0;

	vertices[ny * hw + nx] = 0;
	__syncthreads();

	float A = 0.0;
	float B = 0.0;
	float C = 0.0;

	for (int i = 0; i < wsize; i++) {
		for (int j = 0; j < wsize; j++) {
			if (w * (y + i) + x + j < w * h) {
				A += Ix[w * (y + i) + x + j] * Ix[w * (y + i) + x + j];
				B += Iy[w * (y + i) + x + j] * Iy[w * (y + i) + x + j];
				C += Iy[w * (y + i) + x + j] * Ix[w * (y + i) + x + j];
				if (gradient[(y + i) * w + x + j] > mag)
					mag = gradient[(y + i) * w + x + j];
			}
		}
	}

	if (mag > thresMax) {
		vertex = 2;
	}

	float trace = A + B;
	float det = A * B - C * C;
	float R = det - 0.04 * trace * trace;

	if (R > 1000000.0) {
		vertex = 1;
	}

	if (vertex == 1)
		vertices[ny * hw + nx] = vertex;
	else if (vertices[ny * hw + nx] == 0 && vertex == 2)
		vertices[ny * hw + nx] = vertex;

}

__global__ void extractDownKernel(uchar* vertices, int* angle, int w, int h) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= w) goto end;
	
	// Loop vertically down
	for (int y = 0; y < h; y++) {
		if (vertices[w * y + x] == 1 && y != 0) {
			if (vertices[w * y + x + 1] == 1 || vertices[w * (y - 1) + x] == 1 || vertices[w * (y - 1) + x + 1] == 1)
				vertices[w * y + x] = 2;
		}
		else if (vertices[w * y + x] == 2) {
			if (y == 0) vertices[w * y + x] = 0;
			else if (vertices[w * (y - 1) + x] == 0 && vertices[w * (y - 1) + x - 1] == 0 && vertices[w * (y - 1) + x + 1] == 0)
				vertices[w * y + x] = 0;
		}
		__syncthreads();
	}

end:

}

__global__ void extractUpKernel(uchar* vertices, int* angle, int* vcount, int w, int h) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= w) goto end;

	// Loop vertically down
	for (int y = h - 1; y >= 0; y--) {
		if (vertices[w * y + x] == 1 && y != 0) {
			if (vertices[w * y + x + 1] == 1 || vertices[w * (y + 1) + x] == 1 || vertices[w * (y + 1) + x + 1] == 1)
				vertices[w * y + x] = 2;
		}
		if (vertices[w * y + x] == 2) {
			if (y == h - 1) vertices[w * y + x] = 0;
			else if (vertices[w * (y + 1) + x] == 0 && vertices[w * (y + 1) + x - 1] == 0 && vertices[w * (y + 1) + x + 1] == 0)
				vertices[w * y + x] = 0;
		}
		__syncthreads();
	}

end:

}

__global__ void edgeCountKernel(uchar* vertices, int* angle, int* vcount, int w, int h) {
	
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= w) goto end;
	int prev = 0;

	// Loop vertically down
	for (int y = 0; y < h; y++) {
		vcount[w * y + x] = 0;
		if (vertices[w * y + x] == 1 && y != 0) {
			if (vertices[w * (y - 1) + x] != 0) {
				prev = vcount[w * (y - 1) + x];
				vcount[prev] -= 1;
				vcount[w * y + x] = prev;
			}
			else if (vertices[w * (y - 1) + x + 1] != 0) {
				prev = vcount[w * (y - 1) + x + 1];
				vcount[prev] -= 1;
				vcount[w * y + x] = prev;
			}
			else if (vertices[w * (y - 1) + x - 1] != 0) {
				prev = vcount[w * (y - 1) + x - 1];
				vcount[prev] -= 1;
				vcount[w * y + x] = prev;
			}
			else {
				vcount[w * y + x] = -1;
			}
		}
		else if (vertices[w * y + x] == 2) {
			if (vertices[w * (y - 1) + x] != 0) {
				if (vcount[w * (y - 1) + x] > 0)
					vcount[w * y + x] = vcount[w * (y - 1) + x];
				else
					vcount[w * y + x] = w * (y - 1) + x;
			}
			else if (vertices[w * (y - 1) + x + 1] != 0) {
				if (vcount[w * (y - 1) + x + 1] > 0)
					vcount[w * y + x] = vcount[w * (y - 1) + x + 1];
				else
					vcount[w * y + x] = w * (y - 1) + x + 1;
			}
			else if (vertices[w * (y - 1) + x - 1] != 0) {
				if (vcount[w * (y - 1) + x - 1] > 0)
					vcount[w * y + x] = vcount[w * (y - 1) + x - 1];
				else
					vcount[w * y + x] = w * (y - 1) + x - 1;
			}
		}
		__syncthreads();
	}

end:

}

__global__ void highlightKernel(uchar* frame, uchar* vertices, int* vcount, int w, int h) {

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int nx = (int) (x / wsize);
	int ny = (int) (y / wsize);
	int nw = w / wsize;
	int pixel = y * w * 3 + 3 * x;

	int v = vcount[ny * nw + nx];
	if (v > 0)
		v = vcount[v];
	v = -v;

//	if (v >= 2)
//		frame[pixel + 1] = 255;
	
//	frame[pixel] = 0;
//	frame[pixel + 1] = 0;
//	frame[pixel + 2] = 0;
	
	if (vertices[ny * nw + nx] == 1)
		frame[pixel + 1] = 255;
	else if (vertices[ny * nw + nx] == 2)
		frame[pixel + 2] = 255;
	
}

__global__ void upScaleKernel(uchar* frame, uchar* enlarged, int w, int h, int nw, int nh){
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// New Coordinates
	int x = bx * blockDim.x + tx;
	int y = by * blockDim.y + ty;
	int pixel = y * nw * 3 + 3 * x;

	// Scale Ratio
	float sx = ((float) w) / nw;
	float sy = ((float) h) / nh;

	// Original Coordinates
	int xor = sx * x;
	int yor = sy * y;
	float xo = sx * x;
	float yo = sy * y;

	float dx = xo - (float) ((int) (sx * x));
	float dy = yo - (float) ((int) (sy * y));

	int R = 0;
	int G = 0;
	int B = 0;

	R += frame[yor * w * 3 + 3 * xor] * (1 - dx) * (1 - dy);
	G += frame[yor * w * 3 + 3 * xor + 1] * (1 - dx) * (1 - dy);
	B += frame[yor * w * 3 + 3 * xor + 2] * (1 - dx) * (1 - dy);

	R += frame[yor * w * 3 + 3 * (xor + 1)] * (dx) * (1 - dy);
	G += frame[yor * w * 3 + 3 * (xor + 1) + 1] * (dx) * (1 - dy);
	B += frame[yor * w * 3 + 3 * (xor + 1) + 2] * (dx) * (1 - dy);

	R += frame[(yor + 1) * w * 3 + 3 * (xor)] * (1 - dx)* (dy);
	G += frame[(yor + 1) * w * 3 + 3 * (xor) + 1] * (1 - dx)* (dy);
	B += frame[(yor + 1) * w * 3 + 3 * (xor) + 2] * (1 - dx)* (dy);

	R += frame[(yor + 1) * w * 3 + 3 * (xor + 1)] * (dx)* (dy);
	G += frame[(yor + 1) * w * 3 + 3 * (xor + 1) + 1] * (dx)* (dy);
	B += frame[(yor + 1) * w * 3 + 3 * (xor + 1) + 2] * (dx)* (dy);
	
	enlarged[pixel] = R;
	enlarged[pixel + 1] = G;
	enlarged[pixel + 2] = B;

	/*
	int a = x + y * w, b = a + 1, c = a + w, d = c + 1;

	enlarged[(i + j*nw) * 3] = frame[(x + y*w) * 3];
	enlarged[(i + j*nw) * 3 + 1] = frame[(x + y*w) * 3 + 1];
	enlarged[(i + j*nw) * 3 + 2] = frame[(x + y*w) * 3 + 2];
	
	enlarged[(i + j*nw) * 3] = frame[a * 3] * (1 - dx)*(1 - dy) + frame[b * 3] * dx*(1 - dy) + frame[c * 3] * dy*(1 - dx) + frame[d * 3] * dx*dy;
	enlarged[(i + j*nw) * 3 + 1] = frame[a * 3 + 1] * (1 - dx)*(1 - dy) + frame[b * 3 + 1] * dx*(1 - dy) + frame[c * 3 + 1] * dy*(1 - dx) + frame[d * 3 + 1] * dx*dy;
	enlarged[(i + j*nw) * 3 + 2] = frame[a * 3 + 2] * (1 - dx)*(1 - dy) + frame[b * 3 + 2] * dx*(1 - dy) + frame[c * 3 + 2] * dy*(1 - dx) + frame[d * 3 + 2] * dx*dy;

	// w/ optimization, since ever pix in ABCD only needs 4 reads from frame
	// every 4 corner sets bounds for block
	// BLOCK_SIZE.x = 1/xr, BLOCK_SIZE.y = 1/yr
	
	__shared__ uchar samples[4*3]; //ABCD * RGB
	for(int i=0;i<3;i++){
	samples[i] = frame[a*3+i];
	samples[3+i] = frame[b*3+i];
	samples[6+i] = frame[c*3+i];
	samples[9+i] = frame[d*3+i];
	}
	__syncthreads();

	enlarged[(i+j*nw)*3] = samples[0]*(1-dx)*(1-dy) + samples[3]*dx*(1-dy) + samples[6]*dy*(1-dx) + samples[9]*dx*dy;
	enlarged[(i+j*nw)*3+1] = samples[1]*(1-dx)*(1-dy) + samples[4]*dx*(1-dy) + samples[7]*dy*(1-dx) + samples[10]*dx*dy;
	enlarged[(i+j*nw)*3+2] = samples[2]*(1-dx)*(1-dy) + samples[5]*dx*(1-dy) + samples[8]*dy*(1-dx) + samples[11]*dx*dy;
	*/

}

int main(int argc, uchar** argv) {

	// Function to perform
	// 0 - Test Cuda Device
	// 10 - Gaussian Blur
	// 20 - Edge Detection
	// 30 - Upscale to 1080P
	// 100 - Detect Polygons
	// 200 - Detect Polygons (Image Input)
	int func = 30;

	// Magnitude Thresholds (For Edge Detection Kernel Only)
	int thresMin = 150;
	int thresMax = 300;

	// New Height and Width (For Upscaling Kernel Only)
	int nw = 1920;
	int nh = 1080;
	int res;

	if (func == 30) {
		res = upScale(nw, nh);
		return res;
	}
	else if (func == 200) {
		res = processImage(thresMin, thresMax);
		return res;
	}

	// Opens video file for reading
	VideoCapture cap("C:\\Users\\Edward\\Videos\\in.mp4");
	if (!cap.isOpened()){
		cout << "Cannot open the video file" << endl;
		getchar();
		return -1;
	}

	// Retrieve video properties
	int count = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
	int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("Video Read Success\n");
	printf("Video Properties:\n Frame count: %d\n Resolution: %d x %d\n\n", count, width, height);

	// Allocate heap memory for frames
	Mat* frames = new Mat[count];
	uchar** cframe = new uchar*[count];
	for (int i = 0; i < count; i++) {
		cframe[i] = new uchar[height * width * 3];
	}

	// Extract frames from video file
	bool success;
	for (int i = 0; i < count; i++) {
		cap.set(CV_CAP_PROP_POS_FRAMES, i);
		success = cap.read(frames[i]);
		if (!success){
			cout << "Cannot read frame: " << i << endl;
			abort();
		}
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width * 3; k++)
				cframe[i][j * width * 3 + k] = *(frames[i].ptr(j) + k);
		}

		printf("\r Extracting Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
	}
	printf("\n Successfully Extracted All Frames\n\n");

	// Call Kernel Wrapper
	cudaWrapper(&cframe, count, width, height, func, thresMin, thresMax, nw, nh);

	// Remap Frames
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width * 3; k++) {
				*(frames[i].ptr(j) + k) = cframe[i][j * width * 3 + k];
			}
		}
	}

	// Write output video file
	Size2i vsize(width, height);
	VideoWriter outvid;
	success = outvid.open("C:\\Users\\Edward\\Videos\\out.mpg", CV_FOURCC('M', 'P', 'E', 'G'), cap.get(CV_CAP_PROP_FPS), vsize, true);
	if (!success) {
		cout << "Failed to Create Output Video File" << endl;
	}
	for (int i = 0; i < count; i++) {
		outvid.write(frames[i]);
		printf("\r Writing Output Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
	}

	// Heap memory deallocation
	delete[] frames;
	for (int i = 0; i < count; i++)
		delete[] cframe[i];
	delete[] cframe;

	printf("\n\nVideo Write Completed");
	getchar();


	return 0;
}

int upScale(int nw, int nh) {

	// Opens video file for reading
	VideoCapture cap("C:\\Users\\Edward\\Videos\\in.mp4");
	if (!cap.isOpened()){
		cout << "Cannot open the video file" << endl;
		getchar();
		return -1;
	}

	// Retrieve video properties
	int count = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
	int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("Video Read Success\n");
	printf("Video Properties:\n Frame count: %d\n Resolution: %d x %d\n\n", count, width, height);

	// Allocate heap memory for frames
	Mat* frames = new Mat[count];
	uchar** cframe = new uchar*[count];
	for (int i = 0; i < count; i++) {
		cframe[i] = new uchar[height * width * 3];
	}

	// Single Large Frame to Write
	Mat* nframe_mat = new Mat[1];
	VideoCapture capl("C:\\Users\\Edward\\Videos\\1080p.mp4");
	capl.set(CV_CAP_PROP_POS_FRAMES, 0);
	capl.read(nframe_mat[0]);
	uchar* nframe = new uchar[nh * nw * 3];

	// Extract frames from video file
	bool success;
	for (int i = 0; i < count; i++) {
		cap.set(CV_CAP_PROP_POS_FRAMES, i);
		success = cap.read(frames[i]);
		if (!success){
			cout << "Cannot read frame: " << i << endl;
			abort();
		}
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width * 3; k++)
				cframe[i][j * width * 3 + k] = *(frames[i].ptr(j) + k);
		}
		printf("\r Extracting Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
	}
	printf("\n Successfully Extracted All Frames\n\n");
	cudaError_t cudaStatus = cudaSuccess;
	
	// Write output video file
	Size2i vsize(nw, nh);
	VideoWriter outvid;
	success = outvid.open("C:\\Users\\Edward\\Videos\\outlarge.mpg", CV_FOURCC('M', 'P', 'E', 'G'), cap.get(CV_CAP_PROP_FPS), vsize, true);
	if (!success) {
		cout << "Failed to Create Output Video File" << endl;
	}

	for (int i = 0; i < count; i++) {
		
		cudaStatus = kernelCall_UpScale(&(cframe[i]), &nframe, width, height, nw, nh);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kernelCall failed at frame %d\n", i);
			fprintf(stderr, "Last Error: %s\n", cudaGetErrorString(cudaStatus));
			getchar();
			return 1;
		}

		// Remap Frames
		for (int j = 0; j < nh; j++) {
			for (int k = 0; k < nw * 3; k++) {
				*(nframe_mat[0].ptr(j) + k) = nframe[j * nw * 3 + k];
			}
		}

		outvid.write(nframe_mat[0]);
		printf("\r Writing Output Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
	}

	// Heap memory deallocation
	delete[] frames;
	for (int i = 0; i < count; i++)
		delete[] cframe[i];
	delete[] cframe;
	delete[] nframe_mat;
	delete[] nframe;

	printf("\n\nVideo Write Completed");
	getchar();

	return 0;
}

int processImage(int thresMin, int thresMax) {
	Mat image;
	image = imread("C:\\Users\\Edward\\Pictures\\in.jpg", CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int height = image.rows;
	int width = image.cols;

	uchar* frame = new uchar[height * width * 3];
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width * 3; k++)
			frame[j * width * 3 + k] = *(image.ptr(j) + k);
	}

	int w = width; int h = height;
	int hw = w / wsize; int hh = h / wsize;
	int framesize = w * 3 * h * sizeof(uchar);
	cudaError_t cudaStatus;

	// Initialize Mask and Copy to Constant Memory
	int* SobelMask_X = new int[3 * 3];
	initSobel_X(SobelMask_X);
	cudaStatus = cudaMemcpyToSymbol(SO_X, SobelMask_X, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	int* SobelMask_Y = new int[3 * 3];
	initSobel_Y(SobelMask_Y);
	cudaStatus = cudaMemcpyToSymbol(SO_Y, SobelMask_Y, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	// Allocate Gradient Magnitude Structure
	int gradsize = w * h * sizeof(int);
	int vertexsize = hw * hh * sizeof(uchar);
	int anglesize = w * h * sizeof(int);
	int vcountsize = hw * hh * sizeof(int);
	int isize = w * h * sizeof(float);
	int * grad = new int[w * h];
	uchar * vertex = new uchar[hw * hh];
	int * angle = new int[w * h];
	int * vcount = new int[hw * hh];
	int * d_Grad;
	uchar * d_Vertex;
	int * d_Angle;
	int * d_Vcount;
	float * d_Ix;
	float * d_Iy;
	cudaStatus = cudaMalloc((void**)&d_Grad, gradsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Vertex, vertexsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Angle, anglesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Vcount, vcountsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Ix, isize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Iy, isize);
	assert(cudaStatus == cudaSuccess);

	uchar* frame_a = frame;
	uchar* d_frame_a;

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);

	// Dimensions
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(w / 16.0), ceil(h / 9.0), 1);
	dim3 dimGrid_Q(ceil(hw / 16.0), ceil(hh / 9.0), 1);
	dim3 dimBlock_H(32, 1, 1);
	dim3 dimGrid_H(ceil(hw / 32.0), 1, 1);

	sobelOpAngleKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Grad, d_Angle, d_Ix, d_Iy, w, h);
	edgeVertexKernel << <dimGrid_Q, dimBlock >> > (d_frame_a, d_Grad, d_Vertex, d_Angle, d_Ix, d_Iy, hw, hh, thresMin, thresMax);
	extractDownKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, hw, hh);
	extractUpKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
	edgeCountKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
	highlightKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Vertex, d_Vcount, w, h);
	cudaMemcpy(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost);

	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width * 3; k++) {
			*(image.ptr(j) + k) = frame[j * width * 3 + k];
		}
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);
	waitKey(0);

	cudaFree(d_frame_a);
	cudaFree(d_Grad);
	cudaFree(d_Angle);
	cudaFree(d_Vertex);
	cudaFree(d_Vcount);
	cudaFree(d_Ix);
	cudaFree(d_Iy);

	delete[] frame;
	delete[] grad;
	delete[] angle;
	delete[] vertex;
	delete[] vcount;
	delete[] SobelMask_X;
	delete[] SobelMask_Y;

	return 0;
}

int cudaWrapper(uchar*** frame_p, int count, int w, int h, int func, int thresMin, int thresMax, int nw, int nh)
{

	// Check Cuda Device Compatibility
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if (nDevices > 0) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		printf("  Detected Cuda Device: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
	else
		return -1; // No Device Detected

	// Device Status
	cudaError_t cudaStatus;

	// Call Helper Function
	
	if (func == 0)
		cudaStatus = kernelCall(frame_p, count, w, h, func);
	else if (func == 10)
		cudaStatus = kernelCall_Conv(frame_p, count, w, h, func);
	else if (func == 20)
		cudaStatus = kernelCall_Edge(frame_p, count, w, h, thresMin, thresMax);
	else if (func == 100) {
		cudaStatus = kernelCall_Detect(frame_p, count, w, h, thresMin, thresMax, func);
	}
	else {
		cudaStatus = cudaErrorLaunchFailure;
		printf("\n Incorrect Function Number %d\n", func);
	}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelCall failed!");
		fprintf(stderr, "Last Error: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t kernelCall(uchar*** frame_p, int count, int w, int h, int func)
{
	int framesize = w * 3 * h * sizeof(uchar);
	cudaError_t cudaStatus;

	// Restore Frame Pointer
	uchar** frame = *frame_p;

	// Non-default Stream for Async Memory Copies
	cudaStream_t memStream;
	cudaStatus = cudaStreamCreate(&memStream);
	assert(cudaStatus == cudaSuccess);

	// Frames for Interleaved Kernel Calls
	uchar* frame_a;
	uchar* frame_b;
	uchar* d_frame_a;
	uchar* d_frame_b;

	// Set Default GPU for Execution
	cudaSetDevice(0);

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_b, framesize);
	assert(cudaStatus == cudaSuccess);

	// Copy first frame from host memory to GPU buffers.
	frame_a = *frame;
	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);

	// Dimensions (Simplified Assumption 16 x 9 Videos)
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(w / 16.0), ceil(h / 9.0), 1);

	// Software Pipelined Kernel Calls
	int state = 0;
	for (int i = 1; i < count; i++) {
		printf("\r Processing Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
		cudaDeviceSynchronize();
		if (state == 0) {
			state = state + 1;
			if (i != 1) cudaMemcpyAsync(frame_b, d_frame_b, framesize, cudaMemcpyDeviceToHost, memStream);
			switch (func) {
			case 0: redFilterKernel << <dimGrid, dimBlock >> > (d_frame_a, w, h);
			default:;
			}
			frame_b = *(frame + i);
			cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
		}
		else {
			state = state - 1;
			cudaMemcpyAsync(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost, memStream);
			switch (func) {
			case 0: redFilterKernel << <dimGrid, dimBlock >> > (d_frame_b, w, h);
			default:;
			}
			frame_a = *(frame + i);
			cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
		}
	}

	// Finalize
	cudaDeviceSynchronize();
	printf("\n Processed All %d Frames\n\n", count);

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpyAsync(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost, memStream);
	cudaMemcpyAsync(frame_b, d_frame_b, framesize, cudaMemcpyDeviceToHost, memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	// Destroy Non-Default Streams
	cudaStatus = cudaStreamDestroy(memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaFree(d_frame_a);
	cudaFree(d_frame_b);

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

cudaError_t kernelCall_Conv(uchar*** frame_p, int count, int w, int h, int func) {

	int framesize = w * 3 * h * sizeof(uchar);
	cudaError_t cudaStatus;

	// Restore Frame Pointer
	uchar** frame = *frame_p;

	// Initialize Mask and Copy to Constant Memory
	float* Mask = new float[5 * 5];
	initGaussMask(Mask);
	cudaStatus = cudaMemcpyToSymbol(M, Mask, 5 * 5 * sizeof(float));
	assert(cudaStatus == cudaSuccess);

	// Non-default Stream for Async Memory Copies
	cudaStream_t memStream;
	cudaStatus = cudaStreamCreate(&memStream);
	assert(cudaStatus == cudaSuccess);

	// Frames for Interleaved Kernel Calls
	uchar* frame_a;
	uchar* frame_b;
	uchar* d_frame_a;
	uchar* d_frame_a_out;
	uchar* d_frame_b;
	uchar* d_frame_b_out;

	// Set Default GPU for Execution
	cudaSetDevice(0);

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_a_out, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_b, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_b_out, framesize);
	assert(cudaStatus == cudaSuccess);

	// Copy first frame from host memory to GPU buffers.
	frame_a = *frame;
	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);

	// Dimensions (Simplified Assumption 16 x 9 Videos)
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(w / 16.0), ceil(h / 9.0), 1);

	// Software Pipelined Kernel Calls
	int state = 0;
	for (int i = 1; i < count; i++) {
		printf("\r Processing Frame %d of %d (%d%% Complete)", i, count, (int) ((100.0 * i) / count));
		cudaDeviceSynchronize();
		if (state == 0) {
			state = state + 1;
			if (i != 1) cudaMemcpyAsync(frame_b, d_frame_b_out, framesize, cudaMemcpyDeviceToHost, memStream);
			switch (func) {
			case 10: gaussianBlurKernel << <dimGrid, dimBlock >> > (d_frame_a, d_frame_a_out, w, h);
			default:;
			}
			frame_b = *(frame + i);
			cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
		}
		else {
			state = state - 1;
			cudaMemcpyAsync(frame_a, d_frame_a_out, framesize, cudaMemcpyDeviceToHost, memStream);
			switch (func) {
			case 10: gaussianBlurKernel << <dimGrid, dimBlock >> > (d_frame_b, d_frame_b_out, w, h);
			default:;
			}
			frame_a = *(frame + i);
			cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
		}
	}

	// Finalize
	cudaDeviceSynchronize();
	printf("\n Processed All %d Frames\n\n", count);

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpyAsync(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost, memStream);
	cudaMemcpyAsync(frame_b, d_frame_b, framesize, cudaMemcpyDeviceToHost, memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	// Destroy Non-Default Streams
	cudaStatus = cudaStreamDestroy(memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaFree(d_frame_a);
	cudaFree(d_frame_b);
	cudaFree(d_frame_a_out);
	cudaFree(d_frame_b_out);

	delete[] Mask;

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

cudaError_t kernelCall_Edge(uchar*** frame_p, int count, int w, int h, int thresMin, int thresMax) {

	// Smooth Frames
	// kernelCall_Conv(frame_p, count, w, h, 10);

	int framesize = w * 3 * h * sizeof(uchar);
	cudaError_t cudaStatus;

	// Restore Frame Pointer
	uchar** frame = *frame_p;

	// Initialize Mask and Copy to Constant Memory
	int* SobelMask_X = new int[3 * 3];
	initSobel_X(SobelMask_X);
	cudaStatus = cudaMemcpyToSymbol(SO_X, SobelMask_X, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	int* SobelMask_Y = new int[3 * 3];
	initSobel_Y(SobelMask_Y);
	cudaStatus = cudaMemcpyToSymbol(SO_Y, SobelMask_Y, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	// Allocate Gradient Magnitude Structure
	int gradsize = w * h * sizeof(int);
	int * grad = new int[w * h];
	int * d_Grad;
	cudaStatus = cudaMalloc((void**)&d_Grad, gradsize);
	assert(cudaStatus == cudaSuccess);

	// Non-default Stream for Async Memory Copies
	cudaStream_t memStream;
	cudaStatus = cudaStreamCreate(&memStream);
	assert(cudaStatus == cudaSuccess);

	// Frames for Interleaved Kernel Calls
	uchar* frame_a;
	uchar* frame_b;
	uchar* d_frame_a;
	uchar* d_frame_b;

	// Set Default GPU for Execution
	cudaSetDevice(0);

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_b, framesize);
	assert(cudaStatus == cudaSuccess);

	// Copy first frame from host memory to GPU buffers.
	frame_a = *frame;
	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);

	// Dimensions (Simplified Assumption 16 x 9 Videos)
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(w / 16.0), ceil(h / 9.0), 1);

	// Software Pipelined Kernel Calls
	int state = 0;
	for (int i = 1; i < count; i++) {
		cudaDeviceSynchronize();
		printf("\r Processing Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
		if (state == 0) {
			state = state + 1;

			// Current Process
			if (i != 1) cudaMemcpyAsync(frame_b, d_frame_b, framesize, cudaMemcpyDeviceToHost, memStream);

			sobelOpKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Grad, w, h);
			edgeHystKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Grad, w, h, thresMin, thresMax);

			// Next Iteration
			frame_b = *(frame + i);
			cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
		}
		else {
			state = state - 1;
			cudaMemcpyAsync(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost, memStream);

			// Current Process
			sobelOpKernel << <dimGrid, dimBlock >> > (d_frame_b, d_Grad, w, h);
			edgeHystKernel << <dimGrid, dimBlock >> > (d_frame_b, d_Grad, w, h, thresMin, thresMax);

			// Next Iteration
			frame_a = *(frame + i);
			cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
		}
	}

	// Finalize
	cudaDeviceSynchronize();
	printf("\n Processed All %d Frames\n\n", count);

	// Copy output vector from GPU buffer to host memory.
	//cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
	//cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	// Destroy Non-Default Streams
	cudaStatus = cudaStreamDestroy(memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaFree(d_frame_a);
	cudaFree(d_frame_b);
	cudaFree(d_Grad);

	delete [] grad;
	delete [] SobelMask_X;
	delete [] SobelMask_Y;

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

cudaError_t kernelCall_UpScale(uchar** frame_p, uchar** frame_l, int w, int h, int nw, int nh) {

	int framesize = w * 3 * h * sizeof(uchar);
	int nframesize = nw * 3 * nh * sizeof(uchar);
	cudaError_t cudaStatus;

	// Frames for Interleaved Kernel Calls
	uchar* frame_a;
	uchar* d_frame_a;
	uchar* frame_la;
	uchar* d_frame_la;

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_la, nframesize);
	assert(cudaStatus == cudaSuccess);

	// Copy first frame from host memory to GPU buffers.
	frame_a = *frame_p;
	frame_la = *frame_l;

	// Dimensions (Simplified Assumption 16 x 9 Videos)
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(nw / 16.0), ceil(nh / 9.0), 1);

	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);
	assert(cudaStatus == cudaSuccess);
	upScaleKernel << <dimGrid, dimBlock >> > (d_frame_a, d_frame_la, w, h, nw, nh);
	cudaStatus = cudaMemcpy(frame_la, d_frame_la, nframesize, cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);

	cudaFree(d_frame_a);
	cudaFree(d_frame_la);

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

cudaError_t kernelCall_Detect(uchar*** frame_p, int count, int w, int h, int thresMin, int thresMax, int func) {

	//kernelCall_Conv(frame_p, count, w, h, 10);

	int framesize = w * 3 * h * sizeof(uchar);
	int hw = w / wsize; int hh = h / wsize;
	cudaError_t cudaStatus;

	// Restore Frame Pointer
	uchar** frame = *frame_p;

	// Initialize Mask and Copy to Constant Memory
	int* SobelMask_X = new int[3 * 3];
	initSobel_X(SobelMask_X);
	cudaStatus = cudaMemcpyToSymbol(SO_X, SobelMask_X, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	int* SobelMask_Y = new int[3 * 3];
	initSobel_Y(SobelMask_Y);
	cudaStatus = cudaMemcpyToSymbol(SO_Y, SobelMask_Y, 3 * 3 * sizeof(int));
	assert(cudaStatus == cudaSuccess);

	// Allocate Gradient Magnitude Structure
	int gradsize = w * h * sizeof(int);
	int vertexsize = hw * hh * sizeof(uchar);
	int anglesize = w * h * sizeof(int);
	int vcountsize = hw * hh * sizeof(int);
	int isize = w * h * sizeof(float);
	int * grad = new int[w * h];
	uchar * vertex = new uchar[hw * hh];
	int * angle = new int[w * h];
	int * vcount = new int[hw * hh];
	int * d_Grad;
	uchar * d_Vertex;
	int * d_Angle;
	int * d_Vcount;
	float * d_Ix;
	float * d_Iy;
	cudaStatus = cudaMalloc((void**)&d_Grad, gradsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Vertex, vertexsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Angle, anglesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Vcount, vcountsize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Ix, isize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_Iy, isize);
	assert(cudaStatus == cudaSuccess);

	// Non-default Stream for Async Memory Copies
	cudaStream_t memStream;
	cudaStatus = cudaStreamCreate(&memStream);
	assert(cudaStatus == cudaSuccess);

	// Frames for Interleaved Kernel Calls
	uchar* frame_a;
	uchar* frame_b;
	uchar* d_frame_a;
	uchar* d_frame_b;

	// Set Default GPU for Execution
	cudaSetDevice(0);

	// Allocate GPU buffers  .
	cudaStatus = cudaMalloc((void**)&d_frame_a, framesize);
	assert(cudaStatus == cudaSuccess);
	cudaStatus = cudaMalloc((void**)&d_frame_b, framesize);
	assert(cudaStatus == cudaSuccess);

	// Copy first frame from host memory to GPU buffers.
	frame_a = *frame;
	cudaStatus = cudaMemcpy(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice);

	// Dimensions (Simplified Assumption 16 x 9 Videos)
	dim3 dimBlock(16, 9, 1);
	dim3 dimGrid(ceil(w / 16.0), ceil(h / 9.0), 1);
	dim3 dimGrid_Q(ceil(hw / 16.0), ceil(hh / 9.0), 1);
	dim3 dimBlock_H(32, 1, 1);
	dim3 dimGrid_H(ceil(hw / 32.0), 1, 1);

	// Software Pipelined Kernel Calls
	int state = 0;
	for (int i = 1; i < count; i++) {
		cudaDeviceSynchronize();
		printf("\r Processing Frame %d of %d (%d%% Complete)", i, count, (int)((100.0 * i) / count));
		if (state == 0) {
			state = state + 1;

			// Current Process
			if (i != 1) cudaMemcpyAsync(frame_b, d_frame_b, framesize, cudaMemcpyDeviceToHost, memStream);

			sobelOpAngleKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Grad, d_Angle, d_Ix, d_Iy, w, h);
			edgeVertexKernel << <dimGrid_Q, dimBlock >> > (d_frame_a, d_Grad, d_Vertex, d_Angle, d_Ix, d_Iy, hw, hh, thresMin, thresMax);
			extractDownKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, hw, hh);
			extractUpKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
			edgeCountKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
			highlightKernel << <dimGrid, dimBlock >> > (d_frame_a, d_Vertex, d_Vcount, w, h);

			// Next Iteration
			frame_b = *(frame + i);
			cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
		}
		else {
			state = state - 1;
			cudaMemcpyAsync(frame_a, d_frame_a, framesize, cudaMemcpyDeviceToHost, memStream);

			// Current Process
			sobelOpAngleKernel << <dimGrid, dimBlock >> > (d_frame_b, d_Grad, d_Angle, d_Ix, d_Iy, w, h);
			edgeVertexKernel << <dimGrid_Q, dimBlock >> > (d_frame_b, d_Grad, d_Vertex, d_Angle, d_Ix, d_Iy, hw, hh, thresMin, thresMax);
			extractDownKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, hw, hh);
			extractUpKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
			edgeCountKernel << < dimGrid_H, dimBlock_H >> > (d_Vertex, d_Angle, d_Vcount, hw, hh);
			highlightKernel << <dimGrid, dimBlock >> > (d_frame_b, d_Vertex, d_Vcount, w, h);

			// Next Iteration
			frame_a = *(frame + i);
			cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
		}
	}

	// Finalize
	cudaDeviceSynchronize();
	printf("\n Processed All %d Frames\n\n", count);

	// Copy output vector from GPU buffer to host memory.
	//cudaMemcpyAsync(d_frame_a, frame_a, framesize, cudaMemcpyHostToDevice, memStream);
	//cudaMemcpyAsync(d_frame_b, frame_b, framesize, cudaMemcpyHostToDevice, memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	// Destroy Non-Default Streams
	cudaStatus = cudaStreamDestroy(memStream);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaFree(d_frame_a);
	cudaFree(d_frame_b);
	cudaFree(d_Grad);
	cudaFree(d_Angle);
	cudaFree(d_Vertex);
	cudaFree(d_Vcount);
	cudaFree(d_Ix);
	cudaFree(d_Iy);

	delete[] grad;
	delete[] angle;
	delete[] vertex;
	delete[] vcount;
	delete[] SobelMask_X;
	delete[] SobelMask_Y;

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

void initGaussMask(float* Mask) {

	Mask[0] = 2; Mask[1] = 4; Mask[2] = 5; Mask[3] = 4; Mask[4] = 2;
	Mask[5] = 4; Mask[6] = 9; Mask[7] = 12; Mask[8] = 9; Mask[9] = 4;
	Mask[10] = 5; Mask[11] = 12; Mask[12] = 15; Mask[13] = 12; Mask[14] = 5;
	Mask[15] = 4; Mask[16] = 9; Mask[17] = 12; Mask[18] = 9; Mask[19] = 4;
	Mask[20] = 2; Mask[21] = 4; Mask[22] = 5; Mask[23] = 4; Mask[24] = 2;

	for (int i = 0; i < 25; i++) {
		Mask[i] = Mask[i] / 159.0;
	}
	
}

void initSobel_X(int* SO) {

	SO[0] = -1; SO[1] = 0; SO[2] = 1;
	SO[3] = -2; SO[4] = 0; SO[5] = 2;
	SO[6] = -1; SO[7] = 0; SO[8] = 1;
}

void initSobel_Y(int* SO) {

	SO[0] = -1; SO[1] = -2; SO[2] = -1;
	SO[3] = 0; SO[4] = 0; SO[5] = 0;
	SO[6] = 1; SO[7] = 2; SO[8] = 1;
}


